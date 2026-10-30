"""--prefill-mixer-pieces: a chunk's GDN / attention over consecutive pieces, its MoE whole.

End to end on an RTX 2060 (Ornith): the chunk-budget probe fell from 124.5 to 68.3 KiB/token
at 2 pieces, a 19.9k-token prompt went from 7 chunks to 4 (490 -> 649 tok/s), and the
temperature-0 output matched an unpieced run token for token. What the unit level can pin is
the planning those runs relied on:

* the pieces tile the chunk, on the 64-token lattice the GDN snapshot boundaries live on, and
  each continues the one before it (cached_len advances by the pieces already run);
* only the last piece tracks the hybrid-radix snapshot, with the slot the scheduler's build
  already chose -- and the real request's ping-pong state is not advanced a second time;
* anything the proxies cannot represent runs whole.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from freetoken.models.prefill_pieces import plan_prefill_pieces  # noqa: E402


class _Backend:
    def __init__(self):
        self.prepared = []

    def prepare_metadata(self, batch):
        self.prepared.append(batch)
        batch.attn_metadata = ("meta", batch.padded_reqs[0].cached_len, batch.padded_reqs[0].extend_len)


def _batch(extend=1024, cached=0, **over):
    req = SimpleNamespace(
        table_idx=3, uid=7, input_ids=None, cached_len=cached, device_len=cached + extend,
        extend_len=extend, linear_slot_idx=None, mamba_ping_pong=None,
        mamba_next_track_idx=0, mamba_last_track_seqlen=None,
    )
    b = SimpleNamespace(
        is_prefill=True, is_decode=False, spec_verify=False, padded_reqs=[req], reqs=[req],
        mrope_positions=None, fla_metadata=None,
        positions=torch.arange(cached, cached + extend), input_ids=torch.zeros(extend),
        out_loc=torch.arange(extend),
    )
    for k, v in over.items():
        setattr(b, k, v)
    return b, req


def test_pieces_tile_the_chunk_on_the_lattice_and_continue_each_other():
    b, _ = _batch(extend=1000, cached=300)
    backend = _Backend()
    pieces = plan_prefill_pieces(b, 3, backend, "cpu")
    spans = [(s, e) for s, e, _ in pieces]
    assert spans == [(0, 320), (320, 640), (640, 1000)]  # 1000 // 3 = 333 -> 320 on 64s
    for s, e, piece in pieces:
        p = piece.padded_reqs[0]
        assert (p.cached_len, p.extend_len, p.device_len) == (300 + s, e - s, 300 + e)
        assert torch.equal(piece.positions, b.positions[s:e])
        assert torch.equal(piece.out_loc, b.out_loc[s:e])
        assert piece.attn_metadata[1:] == (300 + s, e - s)
    assert len(backend.prepared) == 3
    assert b.padded_reqs[0].cached_len == 300  # the chunk's own request is not touched


def test_pieces_carry_their_rows_of_the_three_axis_rope():
    """An mrope model (vision served) ropes every row at a [t, h, w] position: each piece takes its
    own columns, so an image prompt splits like a text one (the soft tokens are already in the
    stream when the pieces run)."""
    mrope = torch.arange(3 * 1000).view(3, 1000)
    b, _ = _batch(extend=1000, cached=300, mrope_positions=mrope)
    pieces = plan_prefill_pieces(b, 3, _Backend(), "cpu")
    assert [(s, e) for s, e, _ in pieces] == [(0, 320), (320, 640), (640, 1000)]
    for s, e, piece in pieces:
        assert torch.equal(piece.mrope_positions, mrope[:, s:e])


@pytest.mark.parametrize("over, extend, n", [
    ({}, 1024, 1),                                   # not asked
    ({"is_prefill": False}, 1024, 2),                # decode
    ({"spec_verify": True}, 1024, 2),                # MTP verify window
    ({}, 100, 2),                                    # too short to split on 64s
    ({}, 128, 2),                                    # the last piece (64) would not cross a boundary
])
def test_what_the_proxies_cannot_represent_runs_whole(over, extend, n):
    b, _ = _batch(extend=extend, **over)
    assert plan_prefill_pieces(b, n, _Backend(), "cpu") is None


def test_more_than_one_request_runs_whole():
    b, req = _batch()
    b.padded_reqs = [req, req]
    assert plan_prefill_pieces(b, 2, _Backend(), "cpu") is None


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA metadata stages pinned memory")


class _Pool:
    conv_states = torch.zeros(1, 1, 1, 3)  # kernel - 1 = 3


@needs_cuda
def test_only_the_last_piece_tracks_with_the_slot_the_scheduler_chose():
    from freetoken.attention import linear

    b, req = _batch(extend=1024, cached=0)
    req.mamba_ping_pong = (11, 12)
    req.linear_slot_idx = 5
    dev = torch.device("cuda")
    # the scheduler's build: picks ping_pong[0] and advances the index to 1
    b.fla_metadata = linear.build_fla_metadata(b, dev, _Pool())
    assert int(b.fla_metadata.track_dst[0]) == 11
    assert req.mamba_next_track_idx == 1
    boundary = req.mamba_last_track_seqlen

    pieces = plan_prefill_pieces(b, 2, _Backend(), dev, linear_state_pool=_Pool())
    first, last = pieces[0][2], pieces[-1][2]
    assert first.fla_metadata.track_dst is None
    assert int(last.fla_metadata.track_dst[0]) == 11          # the same slot, not ping_pong[1]
    assert last.padded_reqs[0].mamba_last_track_seqlen == boundary  # the same boundary
    assert req.mamba_next_track_idx == 1                       # not advanced a second time
    assert int(first.fla_metadata.fresh_state_indices[0]) == 5  # a fresh sequence zeroes its slot once
    assert last.fla_metadata.fresh_state_indices is None       # the continuation does not
