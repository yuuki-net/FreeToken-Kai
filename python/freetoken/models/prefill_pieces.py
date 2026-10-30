"""Run a prefill chunk's sequence mixers (GDN / attention) in pieces and its MoE whole.

Why: on an offloaded MoE every prefill chunk streams every layer's expert bank to the GPU,
so the number of chunks a prompt is cut into is the number of full bank transfers it pays
(on an RTX 2060, 16.9 GiB per chunk at ~11 GB/s: 1.5 s of a 4.5 s chunk). The chunk width is
capped by the per-token transient, and that cap is set by the mixers, not the experts --
measured on Ornith with 1024 tokens per call:

    GDN        116.5 KiB/token
    attention   79.8 KiB/token
    MoE         16.7 KiB/token

The layer's peak is its largest component, so the GDN decides how wide a chunk may be.

Mixers are sequential in the token dimension and already support resuming from a cached
prefix -- that is what chunked prefill across forwards is -- so a forward can run them over
consecutive pieces of the same chunk, each piece continuing the one before it, and bound
their transient by the piece. The MoE is per-token and cannot tell the difference, so it runs
once over the whole chunk and its bank crosses the bus once per layer. The engine's transient
probe runs a real forward, so it measures the smaller peak on its own and the chunk-width
solver picks a wider chunk: fewer chunks, fewer bank transfers, same arithmetic per token.

Each piece is a shallow copy of the chunk's batch with its own request proxy (cached_len
advanced by the pieces before it) and its own attention and GDN metadata, built by the same
builders the scheduler uses -- so the continuation semantics are the ones already exercised by
chunked prefill, not new ones.

The hybrid-radix snapshot is the one delicate part. The scheduler builds the chunk's GDN
metadata before the forward, and that build both picks the snapshot slot and advances the
request's ping-pong index. A piece must therefore not pick it again: only the last piece
tracks, using the index as it was before the scheduler's build, and the split points are kept
on the 64-token lattice the snapshot boundaries live on, so the last piece's deepest boundary
is exactly the chunk's.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

# The GDN chunk kernel's block size; snapshot boundaries are multiples of it from a forward's
# first token (attention/linear.py _build_track_metadata).
_LATTICE = 64


def plan_prefill_pieces(batch, n: int, attn_backend, device, linear_state_pool=None):
    """[(start, end, piece_batch), ...] covering the chunk, or None when it should run whole.

    Whole when: fewer than 2 pieces asked; not a prefill; an MTP verify window; more than one
    request (the proxies below are per request); or the chunk
    is too short to split on the lattice with a last piece that still crosses a boundary.
    """
    if n < 2 or not batch.is_prefill or getattr(batch, "spec_verify", False):
        return None
    reqs = batch.padded_reqs
    if len(reqs) != 1:
        return None
    r = reqs[0]
    total = r.extend_len
    step = (total // n) // _LATTICE * _LATTICE
    if step < _LATTICE or total - (n - 1) * step <= _LATTICE:
        return None

    from freetoken.attention.linear import build_fla_metadata

    tracked = False
    if linear_state_pool is not None:
        if batch.fla_metadata is None:
            # Built now, on the real request, so the snapshot choice and its ping-pong advance
            # happen exactly once whoever builds first (the probe leaves it to the first GDN
            # layer; the scheduler builds it before the forward).
            batch.fla_metadata = build_fla_metadata(batch, device, linear_state_pool)
        tracked = batch.fla_metadata.track_dst is not None

    bounds = [(i * step, (i + 1) * step) for i in range(n - 1)] + [((n - 1) * step, total)]
    pieces = []
    for idx, (start, end) in enumerate(bounds):
        last = idx == len(bounds) - 1
        proxy = SimpleNamespace(
            table_idx=r.table_idx,
            uid=getattr(r, "uid", None),
            input_ids=getattr(r, "input_ids", None),
            cached_len=r.cached_len + start,
            device_len=r.cached_len + end,
            extend_len=end - start,
            linear_slot_idx=getattr(r, "linear_slot_idx", None),
            # only the last piece may take the chunk's snapshot, and it must take the slot the
            # scheduler's build already chose -- i.e. the index before that build advanced it
            mamba_ping_pong=getattr(r, "mamba_ping_pong", None) if (last and tracked) else None,
            mamba_next_track_idx=(1 - r.mamba_next_track_idx) if (last and tracked)
            else getattr(r, "mamba_next_track_idx", 0),
            mamba_last_track_seqlen=getattr(r, "mamba_last_track_seqlen", None),
            mamba_restore_src=None,
        )
        piece = copy.copy(batch)
        piece.reqs = [proxy]
        piece.padded_reqs = [proxy]
        piece.positions = batch.positions[start:end]
        if getattr(batch, "mrope_positions", None) is not None:
            # the image soft tokens are already in the stream (the embedding runs before the
            # pieces); what the mixers still need is each row's 3-axis rope position
            piece.mrope_positions = batch.mrope_positions[:, start:end]
        piece.input_ids = batch.input_ids[start:end]
        piece.out_loc = batch.out_loc[start:end]
        piece.fla_metadata = None
        if linear_state_pool is not None:
            piece.fla_metadata = build_fla_metadata(piece, device, linear_state_pool)
        attn_backend.prepare_metadata(piece)
        pieces.append((start, end, piece))
    # A second pass over the same rows in the same step -- the MTP draft head fills its own KV
    # for every row of a prefill chunk -- reuses these pieces instead of running whole, which on
    # a chunk this plan made wider would be the step's new transient peak.
    batch.prefill_pieces = pieces
    return pieces


__all__ = ["plan_prefill_pieces"]
