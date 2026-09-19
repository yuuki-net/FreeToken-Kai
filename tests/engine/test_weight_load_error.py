"""Which startup failures are reported as an unreadable checkpoint (WeightLoadError)."""

from __future__ import annotations

import errno

import pytest

pytest.importorskip("torch")

from freetoken.engine.engine import WeightLoadError, _weight_load_context  # noqa: E402
from freetoken.moe.bank_file import BankFileError  # noqa: E402
from freetoken.moe.host_banks import PinFailed  # noqa: E402


def test_a_checkpoint_read_failure_becomes_weight_load_error():
    with pytest.raises(WeightLoadError, match="ValueError: bad header") as exc:
        with _weight_load_context():
            raise ValueError("bad header")
    assert isinstance(exc.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "error",
    [
        BankFileError("--moe-bank-ram: writing /banks/x.ftmb failed: [Errno 28] No space left on device"),
        PinFailed("cudaHostRegister failed for 12.0 GiB"),
        OSError(errno.ENOMEM, "Cannot allocate memory"),
        MemoryError(),
    ],
    ids=["bank-file", "pin", "enomem", "memory"],
)
def test_failures_that_are_not_the_checkpoint_keep_their_type(error):
    with pytest.raises(type(error)) as exc:
        with _weight_load_context():
            raise error
    assert exc.value is error
