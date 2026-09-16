"""Host RAM for the console: the three parts of the bar must add up to the total."""

from __future__ import annotations

from freetoken.webui import hostmem

MEMINFO = """MemTotal:       24000000 kB
MemFree:         1000000 kB
MemAvailable:    3000000 kB
Buffers:           10000 kB
Cached:         20000000 kB
SwapTotal:      32000000 kB
SwapFree:       31000000 kB
Mlocked:          500000 kB
"""


def test_bar_parts_add_up(tmp_path, monkeypatch):
    f = tmp_path / "meminfo"
    f.write_text(MEMINFO)
    monkeypatch.setattr(hostmem, "meminfo", lambda: hostmem.__dict__["_read"](str(f)))
    doc = hostmem.host_memory()
    kib = 1024
    assert doc["total_bytes"] == 24000000 * kib
    assert doc["used_bytes"] == (24000000 - 3000000) * kib
    assert doc["reclaimable_bytes"] == (3000000 - 1000000) * kib
    assert doc["free_bytes"] == 1000000 * kib
    assert doc["used_bytes"] + doc["reclaimable_bytes"] + doc["free_bytes"] == doc["total_bytes"]
    assert doc["swap_used_bytes"] == 1000000 * kib
    assert doc["engine_bytes"] is None  # no engine root given


def test_engine_share_is_the_process_tree(monkeypatch):
    from freetoken.daemon import osproc

    monkeypatch.setattr(osproc, "tree_pids", lambda root: [root, root + 1])
    monkeypatch.setattr(osproc, "read_pss_bytes", lambda pid: 100)
    doc = hostmem.host_memory(4242)
    assert doc["engine_bytes"] == 200


def test_missing_meminfo_gives_nothing(monkeypatch):
    monkeypatch.setattr(hostmem, "meminfo", lambda: {})
    assert hostmem.host_memory() is None
