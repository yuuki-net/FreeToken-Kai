"""What a --moe-bank-ram bank sits on, read from a fake /proc and /sys.

The machine this is shaped after: an Intel board with the model on a Gen3 NVMe in a chipset
M.2 (PCH root port 00:1d.0), one GPU on the CPU's x16 (00:01.0) and one in a chipset x4 slot
(00:1c.4) -- the layout where --pp-size 2 makes the bank reads and rank 1's traffic share DMI.
"""

from __future__ import annotations

import os

import pytest

from freetoken.moe import disk_probe as dp

GiB = 2**30

NVME = "devices/pci0000:00/0000:00:1d.0/0000:3d:00.0/nvme/nvme0/nvme0n1"
SATA = "devices/pci0000:00/0000:00:17.0/ata1/host0/target0:0:0/0:0:0:0/block/sda"
HYPERV = "devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/fd1d/host0/target0:0:0/0:0:0:5/block/sdf"
USB = "devices/pci0000:00/0000:00:14.0/usb2/2-1/2-1:1.0/host6/target6:0:0/6:0:0:0/block/sdb"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _link(link, target_abs):
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(os.path.relpath(target_abs, os.path.dirname(link)), link)


def _pci(sys, path, vendor, device, cls="0x010802", speed=None, width=None, max_speed=None, max_width=None):
    d = os.path.join(sys, path)
    _write(os.path.join(d, "vendor"), f"0x{vendor:04x}\n")
    _write(os.path.join(d, "device"), f"0x{device:04x}\n")
    _write(os.path.join(d, "class"), cls + "\n")
    if speed:
        _write(os.path.join(d, "current_link_speed"), speed + "\n")
        _write(os.path.join(d, "current_link_width"), f"{width}\n")
        _write(os.path.join(d, "max_link_speed"), (max_speed or speed) + "\n")
        _write(os.path.join(d, "max_link_width"), f"{max_width or width}\n")
    _link(os.path.join(sys, "bus/pci/devices", os.path.basename(path)), d)


def _disk(sys, path, dev, rot=0, ra=8192, vendor=None, model=None, partition=None):
    d = os.path.join(sys, path)
    _write(os.path.join(d, "queue/read_ahead_kb"), f"{ra}\n")
    _write(os.path.join(d, "queue/rotational"), f"{rot}\n")
    if vendor:
        _write(os.path.join(d, "device/vendor"), vendor + "\n")
        _write(os.path.join(d, "device/model"), model + "\n")
    _link(os.path.join(sys, "block", os.path.basename(d)), d)
    if partition:
        pname, pdev = partition
        p = os.path.join(d, pname)
        _write(os.path.join(p, "partition"), "2\n")
        _link(os.path.join(sys, "dev/block", pdev), p)
    else:
        _link(os.path.join(sys, "dev/block", dev), d)
    return d


MOUNTINFO = (
    "29 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
    "40 29 0:84 / /mnt/c rw,noatime - 9p C:\\134 rw,aname=drvfs;path=C:\\;uid=1000\n"
    "41 29 0:31 / /mnt/my\\040disk rw - ext4 /dev/sda1 rw\n"
)


@pytest.fixture
def tree(tmp_path):
    sys, proc = str(tmp_path / "sys"), str(tmp_path / "proc")
    _pci(sys, "devices/pci0000:00/0000:00:1d.0", 0x8086, 0x7AB0, cls="0x060400")
    _pci(sys, "devices/pci0000:00/0000:00:1d.0/0000:3d:00.0", 0x144D, 0xA808,
         speed="8.0 GT/s PCIe", width=4, max_speed="16.0 GT/s PCIe", max_width=4)
    _disk(sys, NVME, "259:0", partition=("nvme0n1p2", "259:2"))
    _pci(sys, "devices/pci0000:00/0000:00:01.0", 0x8086, 0xA70D, cls="0x060400")
    _pci(sys, "devices/pci0000:00/0000:00:01.0/0000:01:00.0", 0x10DE, 0x2504, cls="0x030000")
    _pci(sys, "devices/pci0000:00/0000:00:1c.4", 0x8086, 0x7ABC, cls="0x060400")
    _pci(sys, "devices/pci0000:00/0000:00:1c.4/0000:05:00.0", 0x10DE, 0x2504, cls="0x030000")
    _write(os.path.join(proc, "self/mountinfo"), MOUNTINFO)
    _write(os.path.join(proc, "sys/kernel/osrelease"), "6.8.0-45-generic\n")
    _write(os.path.join(proc, "meminfo"),
           "MemTotal:       65000000 kB\nMemAvailable:   62000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
    _write(os.path.join(proc, "self/limits"),
           "Limit                     Soft Limit           Hard Limit           Units\n"
           "Max locked memory         8388608              8388608              bytes\n")
    return sys, proc


# ----- mounts -------------------------------------------------------------------------
def test_the_longest_mount_point_wins_and_escapes_are_undone():
    mounts = dp.parse_mountinfo(MOUNTINFO)
    assert dp.mount_of("/home/u/.cache/freetoken", mounts).dev == "259:2"
    assert dp.mount_of("/mnt/c/models", mounts).fstype == "9p"
    assert dp.mount_of("/mnt/my disk/bank", mounts).source == "/dev/sda1"
    assert dp.mount_of("/mnt/cache", mounts).mountpoint == "/"  # a prefix of the name is not a parent


def test_a_windows_drive_under_wsl_is_refused_by_name():
    mounts = dp.parse_mountinfo(MOUNTINFO)
    level, why = dp.filesystem_verdict(dp.mount_of("/mnt/c/x", mounts))
    assert level == "bad" and "drvfs" in why


@pytest.mark.parametrize("fstype,level", [
    ("ext4", "ok"), ("xfs", "ok"), ("tmpfs", "bad"), ("nfs4", "bad"), ("cifs", "bad"),
    ("overlay", "warn"), ("fuse.sshfs", "bad"), ("fuse.foo", "warn"), ("zfs", "warn"),
])
def test_filesystem_verdicts(fstype, level):
    assert dp.filesystem_verdict(dp.Mount("/", fstype, "x", "0:1"))[0] == level


# ----- devices ------------------------------------------------------------------------
def test_an_nvme_partition_resolves_to_its_disk_link_and_chipset(tree):
    sys, _ = tree
    d = dp.block_device("259:2", "/dev/nvme0n1p2", sys)
    assert (d.name, d.partition, d.transport, d.virtual, d.rotational) == ("nvme0n1", "nvme0n1p2", "nvme", False, False)
    assert [b for b, _ in d.pci_chain] == ["0000:00:1d.0", "0000:3d:00.0"]
    assert d.link.describe() == "PCIe Gen3 x4 (device supports Gen4 x4)"
    assert dp.upstream_kind(d.pci_chain) == "chipset"


def test_which_gpu_shares_the_chipset_uplink(tree):
    sys, _ = tree
    d = dp.block_device("259:2", "", sys)
    gpus = dict(dp.nvidia_gpus(sys))
    assert set(gpus) == {"0000:01:00.0", "0000:05:00.0"}
    assert dp.upstream_kind(gpus["0000:01:00.0"]) == "cpu"
    assert dp.upstream_kind(gpus["0000:05:00.0"]) == "chipset"
    # different root ports, so no common bridge: sharing DMI is the chipset classification's job
    assert dp.shared_upstream(d.pci_chain, gpus["0000:05:00.0"]) is None


def test_a_common_bridge_is_named():
    a = [("0000:00:01.1", "/a"), ("0000:02:00.0", "/b"), ("0000:03:00.0", "/c")]
    b = [("0000:00:01.1", "/a"), ("0000:02:00.0", "/b"), ("0000:04:00.0", "/d")]
    assert dp.shared_upstream(a, b) == "0000:02:00.0"


def test_device_mapper_is_followed_to_the_drive(tree, tmp_path):
    sys, _ = tree
    dm = os.path.join(sys, "devices/virtual/block/dm-0")
    _write(os.path.join(dm, "queue/read_ahead_kb"), "128\n")
    _link(os.path.join(dm, "slaves/nvme0n1p2"), os.path.join(sys, NVME, "nvme0n1p2"))
    _link(os.path.join(sys, "dev/block/253:0"), dm)
    d = dp.block_device("253:0", "/dev/mapper/root", sys)
    assert d.stacked_on == ["dm-0"] and d.name == "nvme0n1" and d.transport == "nvme"
    # the window that applies is the dm device's own, not the drive's
    assert dp.readahead("253:0", sys)[0] == 128


def test_a_hyperv_disk_is_virtual_whatever_its_rotational_flag_says(tmp_path):
    sys = str(tmp_path)
    _disk(sys, HYPERV, "8:80", rot=1, vendor="Msft    ", model="Virtual Disk    ")
    d = dp.block_device("8:80", "/dev/sdf", sys)
    assert (d.transport, d.virtual, d.model) == ("hyperv", True, "Virtual Disk")


def test_sata_and_usb_are_told_apart(tmp_path):
    sys = str(tmp_path)
    _disk(sys, SATA, "8:0")
    _disk(sys, USB, "8:16")
    assert dp.block_device("8:0", "", sys).transport == "sata"
    assert dp.block_device("8:16", "", sys).transport == "usb"


def test_an_anonymous_device_falls_back_to_the_mount_source(tree):
    sys, _ = tree
    _link(os.path.join(sys, "class/block/nvme0n1p2"), os.path.join(sys, NVME, "nvme0n1p2"))
    assert dp.block_device("0:45", "/dev/nvme0n1p2", sys).name == "nvme0n1"


# ----- readahead ----------------------------------------------------------------------
def test_readahead_through_a_partition_names_a_file_that_exists(tree):
    """normpath turned /sys/dev/block/259:2/../queue into /sys/dev/block/queue."""
    sys, _ = tree
    kb, where = dp.readahead("259:2", sys)
    assert kb == 8192
    assert where == os.path.join(sys, "block/nvme0n1/queue/read_ahead_kb")
    assert os.path.exists(where)


@pytest.mark.parametrize("widest_bytes,expected", [
    (1600 * 1024, 256),     # Flash-Next: measured optimum 256
    (2880 * 2880, 2048),    # gpt-oss-120b, 7.91 MiB: measured optimum 2048
    (1024 * 1024, 256),     # Ornith
    (1, 16),
])
def test_recommended_window_lands_on_the_measured_optima(widest_bytes, expected):
    assert dp.recommend_readahead_kb(widest_bytes) == expected


def test_set_readahead_writes_and_reports_refusal(tmp_path):
    f = tmp_path / "read_ahead_kb"
    f.write_text("8192\n")
    assert dp.set_readahead(str(f), 256) == (True, "")
    assert f.read_text().strip() == "256"
    ok, why = dp.set_readahead(str(tmp_path / "missing" / "read_ahead_kb"), 256)
    assert not ok and why


# ----- memory -------------------------------------------------------------------------
def test_meminfo_and_memlock(tree, tmp_path):
    _, proc = tree
    mem = dp.meminfo(proc)
    assert mem["MemTotal"] == 65000000 * 1024
    assert dp.memlock_limit(proc) == (8388608, 8388608)
    _write(os.path.join(proc, "self/limits"), "Max locked memory         unlimited            unlimited            bytes\n")
    assert dp.memlock_limit(proc) == (None, None)


@pytest.fixture
def native(monkeypatch):
    """A host whose CUDA does not cap pinning (plain Linux), whatever this test runs on."""
    monkeypatch.setattr(dp, "is_wsl", lambda proc="/proc": False)
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)


def test_auto_matches_the_measured_64gb_configuration(native):
    """48G on a 64 GB host with --pp-size 2 was the configuration docs/bank-ram.md measured."""
    mem = {"MemTotal": 62 * GiB, "MemAvailable": 60 * GiB}
    auto = dp.auto_bank_ram(mem, 2)
    assert 47 * GiB < auto.total_bytes < 49 * GiB
    assert auto.nonbank == 9 * GiB
    assert "MemAvailable 60.0 GiB" in auto.reason() and "2 ranks" in auto.reason()
    from freetoken.moe.bank_disk import parse_size

    assert abs(parse_size(auto.as_flag()) - auto.total_bytes) < 0.01 * GiB


def test_auto_leaves_the_headroom_floor_on_a_small_host(native):
    auto = dp.auto_bank_ram({"MemTotal": 23 * GiB, "MemAvailable": 20 * GiB}, 1)
    assert auto.headroom == 2 * GiB
    assert auto.total_bytes == int(20 * GiB - 4.5 * GiB - 2 * GiB)


def test_auto_refuses_when_nothing_is_left(native):
    with pytest.raises(ValueError, match="pass a size"):
        dp.auto_bank_ram({"MemTotal": 16 * GiB, "MemAvailable": 7 * GiB}, 1)
    with pytest.raises(ValueError, match="not readable"):
        dp.auto_bank_ram({}, 1)


# ----- storage warnings ---------------------------------------------------------------
def test_no_warning_for_an_nvme(tree, tmp_path):
    sys, proc = tree
    assert dp.storage_warnings(str(tmp_path), proc, sys) == []


def test_a_drvfs_bank_directory_is_warned_about(tree, tmp_path):
    sys, proc = tree
    _write(os.path.join(proc, "self/mountinfo"),
           MOUNTINFO + f"50 29 0:84 / {tmp_path} rw - 9p C:\\134 rw,aname=drvfs;path=C:\\\n")
    (msg,) = dp.storage_warnings(str(tmp_path / "bankmap"), proc, sys)
    assert "drvfs" in msg and "--moe-bank-dir" in msg


def test_sata_rotational_and_usb_are_warned_about_but_a_virtual_disk_is_not(tmp_path):
    sys, proc = str(tmp_path / "sys"), str(tmp_path / "proc")
    _disk(sys, SATA, "8:0")
    _disk(sys, USB, "8:16")
    _disk(sys, HYPERV, "8:80", rot=1, vendor="Msft", model="Virtual Disk")
    sda2 = os.path.join(sys, "devices/virtual/block/sdz")  # a spinning disk
    _write(os.path.join(sda2, "queue/rotational"), "1\n")
    _link(os.path.join(sys, "dev/block/8:32"), sda2)
    for dev, needle in (("8:0", "SATA"), ("8:16", "USB"), ("8:32", "rotating"), ("8:80", None)):
        _write(os.path.join(proc, "self/mountinfo"), f"1 0 {dev} / / rw - ext4 /dev/x rw\n")
        got = dp.storage_warnings(str(tmp_path), proc, sys)
        assert (needle in got[0]) if needle else got == [], (dev, got)


# ----- other processes and the benchmark ----------------------------------------------
def test_mapped_by_finds_a_server_holding_the_file(tmp_path):
    proc = tmp_path / "proc"
    bank = tmp_path / "bank.rank0of1.ftmb"
    bank.write_bytes(b"x")
    _write(str(proc / "4242/maps"), f"7f00-7f10 r--s 00000000 08:50 12 {bank}\n")
    _write(str(proc / "4242/comm"), "python3\n")
    _write(str(proc / "4343/maps"), "7f00-7f10 r--p 00000000 08:50 13 /usr/lib/libc.so.6\n")
    found, unreadable = dp.mapped_by(str(bank), str(proc))
    assert found == [(4242, "python3")] and unreadable == 0


def test_random_row_read_reads_without_the_page_cache(tmp_path):
    f = tmp_path / "bank"
    f.write_bytes(os.urandom(4 << 20))
    try:
        gbs = dp.random_row_read(str(f), 300_000, 2, 0.2)
    except OSError as exc:  # a TMPDIR on tmpfs refuses O_DIRECT, which is the point of refusing
        pytest.skip(f"O_DIRECT unavailable here: {exc}")
    assert gbs > 0


def test_physical_cores_counts_core_pairs(tmp_path):
    sys = str(tmp_path)
    for cpu, (pkg, core) in enumerate([(0, 0), (0, 0), (0, 1), (0, 1), (0, 2)]):
        _write(os.path.join(sys, f"devices/system/cpu/cpu{cpu}/topology/physical_package_id"), f"{pkg}\n")
        _write(os.path.join(sys, f"devices/system/cpu/cpu{cpu}/topology/core_id"), f"{core}\n")
    assert dp.physical_cores(sys) == 3


# ----- WSL2: the Windows drive under the virtual disk ---------------------------------
REG = (
    "\r\nHKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\r\n"
    "\r\nHKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\\{1bd2}\r\n"
    "    DistributionName    REG_SZ    Ubuntu\r\n"
    "    BasePath    REG_SZ    \\\\?\\D:\\wsl\\{1bd2}\r\n"
    "    VhdFileName    REG_SZ    ext4.vhdx\r\n"
    "\r\nHKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\\{1eef}\r\n"
    "    DistributionName    REG_SZ    rancher-desktop\r\n"
    "    BasePath    REG_SZ    C:\\Users\\u\\rd\r\n"
)


def test_the_registry_names_this_distros_virtual_disk():
    assert dp.parse_reg_lxss(REG, "ubuntu") == (r"D:\wsl\{1bd2}", "ext4.vhdx")
    assert dp.parse_reg_lxss(REG, "rancher-desktop") == (r"C:\Users\u\rd", "ext4.vhdx")
    assert dp.parse_reg_lxss(REG, "Debian") is None


def _wsl_proc(tmp_path, drive_dir):
    proc = tmp_path / "proc"
    _write(str(proc / "sys/kernel/osrelease"), "6.18.33.2-microsoft-standard-WSL2\n")
    _write(str(proc / "self/mountinfo"),
           "1 0 8:80 / / rw - ext4 /dev/sdf rw\n"
           f"2 1 0:84 / {drive_dir} rw - 9p D:\\134 rw,aname=drvfs;path=D:\\\n")
    return str(proc)


def test_wsl_host_disk_reads_the_drive_under_the_vhdx(tmp_path, monkeypatch):
    drive = tmp_path / "mnt_d"
    (drive / "wsl" / "{1bd2}").mkdir(parents=True)
    (drive / "wsl" / "{1bd2}" / "ext4.vhdx").write_bytes(b"x" * 4096)
    proc = _wsl_proc(tmp_path, drive)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(dp, "_windows_exe", lambda name: "/fake/" + name)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[0])
        out = REG if cmd[0].endswith("reg.exe") else "Archive, SparseFile\r\n"
        return type("R", (), {"stdout": out})()

    host = dp.wsl_host_disk(proc, run=run)
    assert (host.vhdx, host.drive, host.mount) == (r"D:\wsl\{1bd2}\ext4.vhdx", "D:", str(drive))
    assert host.free_bytes > 0 and host.vhdx_bytes == 4096 and host.sparse is None
    assert calls == ["/fake/reg.exe"]  # the startup path asks no PowerShell
    assert dp.wsl_host_disk(proc, run=run, attributes=True).sparse is True
    assert dp.wsl_host_disk(str(tmp_path / "proc_native"), run=run) is None  # not WSL


def test_host_space_warning_counts_the_windows_drive_not_df(tmp_path):
    drive = tmp_path / "mnt_d"
    drive.mkdir()
    proc = _wsl_proc(tmp_path, drive)
    host = dp.WslHostDisk("Ubuntu", r"D:\wsl\ext4.vhdx", "D:", str(drive), 30 * GiB, 900 * GiB)
    msg = dp.host_space_warning(str(tmp_path / "bankmap" / "bank.ftmb"), 16 * GiB, proc, host=host)
    assert "D: has 30.0 GiB free" in msg and "`df /`" in msg and "--moe-bank-dir" in msg
    assert dp.host_space_warning(str(tmp_path / "bankmap"), 5 * GiB, proc, host=host) is None
    # a path on the drive itself is checked by the ordinary free-space test, not this one
    assert dp.host_space_warning(str(drive / "bank.ftmb"), 16 * GiB, proc, host=host) is None


def test_a_new_bank_is_refused_on_drvfs_and_tmpfs_but_not_on_ext4(tree, tmp_path):
    sys, proc = tree
    assert dp.refuses_new_bank(str(tmp_path / "bankmap"), proc, sys) is None
    _write(os.path.join(proc, "self/mountinfo"),
           MOUNTINFO + f"50 29 0:84 / {tmp_path} rw - 9p C:\\134 rw,aname=drvfs;path=C:\\\n")
    msg = dp.refuses_new_bank(str(tmp_path / "bankmap"), proc, sys)
    assert "not creating a bank file" in msg and "drvfs" in msg
    _write(os.path.join(proc, "self/mountinfo"), MOUNTINFO + f"50 29 0:90 / {tmp_path} rw - tmpfs tmpfs rw\n")
    assert "RAM" in dp.refuses_new_bank(str(tmp_path / "bankmap"), proc, sys)


def test_under_wsl_the_pin_budget_limits_registration_not_residency(monkeypatch):
    """Measured on the 2060 host: RAM allows 15.6 GiB of 23.5 and the pin budget 9.4 GiB.

    The budget used to size the resident rows, which cost a host with plenty of RAM the rows the
    CPU executor wanted (and refused the flag outright at 1 GiB, FreeToken-Kai#2). Residency is
    RAM's decision; the budget says how many layers the GPU will be able to address, and the
    message has to make that split visible."""
    monkeypatch.setattr(dp, "is_wsl", lambda proc="/proc": True)
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    mem = {"MemTotal": int(23.5 * GiB), "MemAvailable": int(22.1 * GiB)}
    auto = dp.auto_bank_ram(mem, 1)
    assert auto.total_bytes == int(22.1 * GiB) - int(4.5 * GiB) - 2 * GiB  # RAM: 15.6 GiB
    why = auto.reason()
    assert "MemAvailable 22.1" in why
    assert "CUDA pin budget 9.4 GiB" in why and "the rest decode on the CPU" in why
    # a budget over the residency has nothing to say
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "64")
    assert "CUDA pin budget" not in dp.auto_bank_ram(mem, 1).reason()


def test_fault_read_copies_through_a_mapping(tmp_path):
    from freetoken.moe import disk_probe as dp

    f = tmp_path / "bank.ftmb"
    f.write_bytes(b"\x01" * (80 << 20))
    gbs, done = dp.fault_read(str(f), seconds=5.0, nbytes=40 << 20, piece=8 << 20)
    assert done == 40 << 20 and gbs > 0
    with pytest.raises(OSError, match="too small"):
        small = tmp_path / "small"
        small.write_bytes(b"\x00" * 4096)
        dp.fault_read(str(small), seconds=1.0)


def test_piece_read_reads_the_range(tmp_path, monkeypatch):
    from freetoken.moe import disk_probe as dp

    monkeypatch.setenv("FREETOKEN_BANK_READ_THREADS", "2")
    monkeypatch.setenv("FREETOKEN_BANK_READ_PIECE_MB", "1")
    f = tmp_path / "bank.ftmb"
    f.write_bytes(os.urandom(16 << 20))
    gbs, how = dp.piece_read(str(f), seconds=5.0, nbytes=6 << 20)
    assert gbs > 0 and how.startswith("2 threads x 1 MiB")
