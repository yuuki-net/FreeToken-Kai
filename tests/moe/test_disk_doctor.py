"""``ft doctor disk``: the prediction arithmetic, and the whole report over a fake host."""

from __future__ import annotations

import json
import os

import pytest

from freetoken.moe import disk_doctor as dd

from .test_disk_probe import tree  # noqa: F401  (fixture)

GiB, MiB = 2**30, 2**20
ROW = 2768240  # Flash-Next: 2.64 MiB per expert


def _flash_next(top_k=10):
    return dd.Shape(name="Qwen3.8-Flash-Next-NVFP4", moe_layers=48, num_experts=512, top_k=top_k,
                    cell_bytes=ROW, widest_row_bytes=1600 * 1024)


def _skewed(layers, experts, hot, a, b):
    """Every layer: the first ``hot`` experts ``a`` routes each, the rest ``b``."""
    return {layer: [a] * hot + [b] * (experts - hot) for layer in layers}


def test_the_prediction_reproduces_the_measured_64gb_disk_term():
    """docs/bank-ram.md / guides 17 §34: Flash-Next at 48G on a 64 GB host, 87.5% coverage,
    5.37 GB/s measured on the drive -> 7.2 ms of disk per step."""
    fit = _skewed(range(48), 512, 387, 226, 100)  # 387*226 / (387*226 + 125*100) = 0.875
    (row,) = dd.predict(_flash_next(), [48 * GiB], 2, 61 * GiB, fit=fit, disk_gbs=5.37, base_step_ms=59.5)
    assert row.hot == 387
    assert row.coverage == pytest.approx(0.875, abs=1e-3)
    assert row.coverage_kind == "in-sample"
    assert 6.5 < row.disk_ms < 8.5
    assert row.step_ms == pytest.approx(59.5 + row.disk_ms)


def test_without_a_histogram_coverage_is_the_residency():
    (row,) = dd.predict(_flash_next(), [24 * GiB], 1, 60 * GiB)
    assert row.coverage_kind == "arbitrary"
    assert row.coverage == pytest.approx(row.residency)
    assert row.disk_ms is None  # no read rate, so no milliseconds


def test_a_cap_that_covers_the_banks_reads_nothing():
    shape = _flash_next()
    (row,) = dd.predict(shape, [shape.bank_bytes], 2, 128 * GiB, disk_gbs=5.0, base_step_ms=50)
    assert row.hot == 512 and row.disk_bytes == 0 and row.step_ms == 50


def test_held_out_coverage_uses_the_fit_for_the_order_only():
    fit = {0: [10, 9, 1, 0]}
    evaluate = {0: [0, 0, 5, 5]}  # a session routed to what the fit thought was cold
    assert dd.coverage(fit, None, 1, 4, 2) == pytest.approx(19 / 20)
    assert dd.coverage(fit, evaluate, 1, 4, 2) == 0.0


def test_page_cache_factor_is_measured_at_the_reference_share_and_one_without_cache():
    assert dd.page_cache_factor(0.0) == 1.0
    assert dd.page_cache_factor(dd.PAGE_CACHE_REF_SHARE) == pytest.approx(dd.PAGE_CACHE_FACTOR)
    assert dd.page_cache_factor(5.0) == pytest.approx(dd.PAGE_CACHE_FACTOR)


def test_default_caps_stay_under_memtotal():
    caps = dd.default_caps(60 * GiB, 64 * GiB, auto=48 * GiB)
    assert 48 * GiB in caps and 60 * GiB in caps and all(c < 64 * GiB for c in caps)
    assert dd.default_caps(100 * GiB, 64 * GiB, None) == [50 * GiB]


def _bank_file(directory, layers=(0, 1), present=(), canonical=None, experts=64):
    """A v2 bank file (every MoE layer in one file) holding ``present`` layers, the rest unwritten."""
    from freetoken.moe.bank_file import BankFile, MappedBankLayout

    lay = MappedBankLayout(
        experts, list(layers),
        [("gate_up", (1280, 1280), "uint8", 1600 * 1024), ("down", (2560, 320), "uint8", 800 * 1024)],
        {"kind": "nvfp4", "kernel": "triton"},
    )
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "bank.ftmb")
    with BankFile.create(path, lay) as f:
        for layer in present:
            f._commit_layer(layer, list(range(experts)), {"gate_up": "0" * 64, "down": "0" * 64})
        if canonical:
            f.set_canonical_for(canonical)
    return path


def test_the_report_over_a_fake_host(tree, tmp_path):  # noqa: F811
    sys, proc = tree
    bank_dir = tmp_path / "bankmap" / "Flash"
    _bank_file(str(bank_dir), present=(0,))
    (bank_dir / "bank.rank0of2.ftmb").write_bytes(b"old")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"layer_range": [0, 2], "decode_freq": [[5] * 16 + [1] * 48] * 2}))
    ns = dd.build_parser("ft doctor disk").parse_args(
        ["--moe-bank-dir", str(bank_dir), "--bench-seconds", "0", "--moe-bank-stats", str(stats),
         "--disk-gbs", "5", "--base-step-ms", "60"]
    )
    out = dd.run(ns, proc, sys)
    assert "2 MoE layers x 64 experts" in out
    assert f"bank file (--moe-bank-dir): {bank_dir}/bank.ftmb, holds layers 0 of 2, nvfp4 / triton" in out
    assert "per-rank bank files from an older build are no longer read" in out
    assert "PCIe Gen3 x4 (device supports Gen4 x4)" in out
    assert "upstream (estimated from the PCI topology): chipset" in out
    assert "GPU 0000:05:00.0" in out and "GPU 0000:01:00.0" not in out
    assert "recommended: 256 kB for a widest row block of 1600 kB" in out
    assert f"echo 256 | sudo tee {sys}/block/nvme0n1/queue/read_ahead_kb" in out
    assert "!! read_ahead_kb 8192 is wider than the widest expert row block" in out
    assert "Gen3" in out.split("Findings")[1]
    assert "memlock (ulimit -l): 8 MiB soft" in out
    assert "--moe-bank-ram auto would choose" in out
    # top-k is unknown without --model: the table still prints, with question marks
    assert "counted on the same histogram that orders the placement" in out
    assert "Prediction per RAM cap" in out


def test_a_gguf_file_gives_the_shape_from_its_tensor_table(tree, tmp_path, monkeypatch):  # noqa: F811
    # FreeToken-Kai#4: --model took only a directory, and a GGUF checkpoint is one file whose layers
    # keep their own ggml types; the expert size is their mean, which --moe-bank-ram solves against
    from freetoken.layers.quantization.ggml_types import row_bytes

    from ._gguf_writer import UD_IQ4_XS_TYPES, write_experts_gguf

    sys, proc = tree
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = str(tmp_path / "Flash-UD-IQ4_XS-00001-of-00001.gguf")
    write_experts_gguf(path, first_block=1, metadata={"general.architecture": "qwen4exp", "qwen4exp.expert_used_count": 2})
    cells = [2 * 256 * row_bytes(int(g), 256) + 256 * row_bytes(int(d), 256) for g, d in UD_IQ4_XS_TYPES]

    shape = dd.Shape()
    dd.shape_from_model(path, shape)
    assert shape.name == "Flash-UD-IQ4_XS-00001-of-00001"
    assert (shape.moe_layers, shape.num_experts, shape.top_k) == (4, 4, 2)
    assert shape.cell_bytes == -(-sum(cells) // len(cells))
    assert shape.widest_row_bytes == max(2 * 256 * row_bytes(int(g), 256) for g, _ in UD_IQ4_XS_TYPES)
    assert dd._largest_file(path) == path

    ns = dd.build_parser("ft doctor disk").parse_args(["--model", path, "--bench-seconds", "0"])
    out = dd.run(ns, proc, sys)
    assert "Flash-UD-IQ4_XS-00001-of-00001: 4 MoE layers x 4 experts, top-2" in out
    assert "from: GGUF tensor table (3 type mixes)" in out
    assert "is not a local directory" not in out


def test_the_report_without_anything_to_go_on(tree, tmp_path, monkeypatch):  # noqa: F811
    sys, proc = tree
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty"))
    ns = dd.build_parser("ft doctor disk").parse_args(["--bench-seconds", "0"])
    out = dd.run(ns, proc, sys)
    assert "no bank file under" in out
    assert "needs the model shape" in out


def test_one_file_describes_every_layer_whichever_ranks_wrote_them(tmp_path):
    path = _bank_file(str(tmp_path), layers=range(4), present=(2, 3))
    shape = dd.Shape()
    state = dd.shape_from_bank(path, shape)
    assert shape.moe_layers == 4 and shape.num_experts == 64
    assert shape.cell_bytes == 2400 * 1024 and shape.widest_row_bytes == 1600 * 1024
    assert state.layers == [2, 3] and state.all_layers == 4 and state.canonical_for is None


def test_a_packed_checkpoint_names_its_own_bank_file(tmp_path):
    from freetoken.moe.bank_pack import PACK_FORMAT, PACK_MANIFEST

    slim = tmp_path / "slim"
    slim.mkdir()
    (slim / PACK_MANIFEST).write_text(json.dumps({"format": PACK_FORMAT, "bank": "bank.ftmb"}))
    path, how = dd.find_bank_file(str(slim), None)
    assert path == str(slim / "bank.ftmb") and "packed" in how
    _bank_file(str(slim), present=(0, 1), canonical=str(slim))
    assert dd.shape_from_bank(path, dd.Shape()).canonical_for == str(slim)
    # --moe-bank-dir still wins, as it does for ft serve
    assert dd.find_bank_file(str(slim), str(tmp_path / "d"))[0] == str(tmp_path / "d" / "bank.ftmb")


def test_an_unreadable_file_is_a_finding_not_a_crash(tmp_path):
    (tmp_path / "bank.ftmb").write_bytes(b"FTMB" + b"\0" * 12)
    state = dd.shape_from_bank(str(tmp_path / "bank.ftmb"), dd.Shape())
    assert state.error


def test_ft_doctor_is_wired_into_the_cli(capsys):
    from freetoken.cli import main

    assert main(["doctor", "--help"]) == 0
    assert "disk" in capsys.readouterr().out
    assert main(["doctor", "nope"]) == 2


def test_under_wsl_the_report_shows_the_windows_drive_and_flags_a_short_one():
    from types import SimpleNamespace

    from freetoken.moe import disk_probe as dp

    storage = SimpleNamespace(mount=SimpleNamespace(mountpoint="/"))
    host = dp.WslHostDisk("Ubuntu", r"C:\wsl\ext4.vhdx", "C:", "/mnt/c", 30 * GiB, 953 * GiB, 133 * GiB, False)
    rep = dd.Report()
    dd._wsl_host(rep, "Bank storage", storage, "/proc", 16 * GiB, host=host)
    text = "\n".join(rep.lines)
    assert r"C:\wsl\ext4.vhdx, 133.0 GiB, not sparse" in text
    assert "C: has 30.0 GiB free of 953 GiB" in text and "not returned to Windows" in text
    (level, msg), = rep.findings
    assert level == "warn" and "16.0 GiB the bank file still needs" in msg
    rep = dd.Report()
    dd._wsl_host(rep, "Bank storage", storage, "/proc", 0, host=dp.WslHostDisk(
        "Ubuntu", r"C:\wsl\ext4.vhdx", "C:", "/mnt/c", 200 * GiB, 953 * GiB, None, True))
    assert rep.findings == [] and "not returned" not in "\n".join(rep.lines)


def _link(**kw):
    from freetoken.moe import gpu_probe as gp

    base = dict(index=1, bus_id="00000000:05:00.0", name="RTX 3060", gen=1, gen_max=4, gen_host=4, width=4, width_max=16)
    base.update(kw)
    return gp.GpuLink(**base)


def test_the_gpu_section_reads_the_link_under_load(tree, tmp_path):  # noqa: F811
    sys, proc = tree
    ns = dd.build_parser("ft doctor disk").parse_args(["--bench-seconds", "0", "--pp-size", "2"])
    out = dd.run(ns, proc, sys, gpu_query=lambda: [_link()], gpu_rate=lambda link, s: (6.3, _link(gen=4)))
    assert "GPU 1 (RTX 3060, 00000000:05:00.0): PCIe Gen1 x4 (GPU Gen4 x16, slot Gen4) as reported now" in out
    assert "during a copy: PCIe Gen4 x4" in out
    assert "pinned host -> GPU: 6.3 GB/s (80% of the link's nominal 7.9 GB/s)" in out
    assert "GPU 1 runs at x4 of its x16 under load" in out.split("Findings")[1]


def test_the_gpu_section_without_a_rate_does_not_judge_the_idle_link(tree, tmp_path):  # noqa: F811
    sys, proc = tree
    ns = dd.build_parser("ft doctor disk").parse_args(["--bench-seconds", "0", "--h2d-seconds", "0"])
    out = dd.run(ns, proc, sys, gpu_query=lambda: [_link()])
    assert "rate not measured (--h2d-seconds 0)" in out
    assert "GPU 1 runs" not in out


def test_parse_nvidia_smi_links():
    from freetoken.moe import gpu_probe as gp

    (a, b) = gp.parse_links("0, 00000000:01:00.0, NVIDIA GeForce RTX 3060, 4, 4, 4, 16, 16\n"
                            "1, 00000000:05:00.0, NVIDIA GeForce RTX 3060, 3, 4, 5, 4, 16\n")
    assert a.describe() == "PCIe Gen4 x16" and a.lane_gbs == pytest.approx(31.5, abs=0.1)
    assert b.describe() == "PCIe Gen3 x4 (GPU Gen4 x16, slot Gen5)"
    assert gp.parse_links("garbage\n[N/A], x\n") == []


def test_prefill_prediction_rereads_the_non_resident_rows_when_the_page_cache_is_short():
    """The reported 64 GB host: Flash-Next, --pp-size 2, 42G cap. The page cache left per rank is
    smaller than a rank's non-resident rows, so each chunk reads them all from the disk again."""
    shape = _flash_next()
    rows = dd.predict(shape, [42 * GiB, shape.bank_bytes], 2, 61 * GiB)
    rep = dd.Report()
    dd._prefill_prediction(rep, rows, 2, shape, 1.0, h2d_gbs=10.0, fault_gbs=0.25)
    out = rep.render()
    short, full = rows
    assert short.page_cache / 2 < short.cold_bytes / 2
    cold = short.cold_bytes / 2
    assert f"{cold / GiB:12.1f}G" in out and f"{cold / 1e9:7.1f}s" in out
    assert "with FREETOKEN_BANK_PREAD=0 the copy faults the rows in through the mapping instead: x 4.0" in out
    (finding,) = [text for level, text in rep.findings if level == "warn"]
    assert "every chunk reads them from the disk again" in finding and "42G" in finding
    # the whole bank resident: nothing to read, only the transfer
    assert full.cold_bytes == 0


def test_prefill_prediction_without_rates_prints_question_marks():
    shape = _flash_next()
    rows = dd.predict(shape, [24 * GiB], 1, 30 * GiB)
    rep = dd.Report()
    dd._prefill_prediction(rep, rows, 1, shape, None, h2d_gbs=None)
    out = rep.render()
    assert "not measured: a complete bank file nobody has mapped is needed; --prefill-read-gbs gives one" in out
    assert "(not measured: --h2d-seconds)" in out
    assert rep.findings == []
