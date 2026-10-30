"""``FT_KERNEL_PROFILE=N``: every N decode steps (verify windows under ``--spec-mtp``), run ONE
step under the torch profiler (CUPTI, CUDA activity only) and log where its GPU time went: a
few buckets (dense GEMMs, the head, experts, GDN, attention, hyper-connections, the rest) and the
top kernels by time. Kernels launched from a replayed CUDA graph are recorded like any other,
so the window graph and the draft-head graphs are covered; the profiled step itself is slower
(the first one by a lot: CUPTI warms up), so it is a breakdown, not a timing. Stops after
``FT_KERNEL_PROFILE_COUNT`` samples (default 3). Each pipeline rank logs its own step.
``FT_KERNEL_PROFILE_PREFILL=1`` also samples prefill chunks (every N-th forward of at least
``FT_KERNEL_PROFILE_MIN_ROWS`` rows, default 0), for a breakdown of where a slow prefill goes;
``FT_KERNEL_PROFILE_PREFILL=only`` samples nothing but those chunks (decode steps would
otherwise use up the sample count before the first long prompt arrives).

The buckets are by kernel name, so a kernel shared by two roles lands in one bucket: the fp8
dense GEMM also serves the lm_head (on the rank that owns it); ``dense`` therefore includes
the head, and the head's share is the kernel's time at the head's shape (count the rows).
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager

import torch

# first match wins: the expert kernels are named first so a dense NVFP4/fp8 GEMM (``_nvfp4_gemm``,
# ``_gemm_kernel``, ``_gemv_splitk``) is not caught by a looser pattern below it
_BUCKETS = (
    ("experts", re.compile(r"moe|ggml|mul_mat|marlin|fast_index_copy|fused_experts|silu_and_mul|moe_align|topk|router|ensure_experts|record_routes", re.I)),
    ("gdn", re.compile(r"fused_recurrent|gated_delta|gdn|conv1d|causal_conv|chunk_", re.I)),
    ("attention", re.compile(r"attn|attention|qsa|flash|index_norm_rope|rope|paged|_gate_", re.I)),
    ("hc_norm", re.compile(r"hc_|rmsnorm|rms_norm|layer_norm|grouped_gemma", re.I)),
    ("dense", re.compile(r"gemm|gemv|splitk|cublas|cutlass|wmma|scaled_mm", re.I)),
    # the dequant scratch of the pre-Ampere W8A16 / NVFP4 dense path shows up here as copies
    ("copy", re.compile(r"memcpy|memset|copy_|cast|index_select|gather|scatter|fill|CatArray", re.I)),
)


class KernelProfiler:
    def __init__(self, every: int, limit: int, log, rank: int = 0, prefill: str = "0", min_rows: int = 0) -> None:
        self.every = max(int(every), 1)
        self.limit = int(limit)
        self.prefill = str(prefill)  # "0" (decode only), "1" (both), "only" (prefill chunks only)
        self.min_rows = int(min_rows)
        self.log = log
        self.rank = rank
        self.steps = 0
        self.samples = 0

    @classmethod
    def from_env(cls, log, rank: int = 0) -> "KernelProfiler | None":
        every = int(os.environ.get("FT_KERNEL_PROFILE", "0") or 0)
        if every <= 0:
            return None
        return cls(
            every, int(os.environ.get("FT_KERNEL_PROFILE_COUNT", "3") or 3), log, rank,
            prefill=os.environ.get("FT_KERNEL_PROFILE_PREFILL", "0") or "0",
            min_rows=int(os.environ.get("FT_KERNEL_PROFILE_MIN_ROWS", "0") or 0),
        )

    def wants(self, is_prefill: bool, spec_verify: bool, rows: int) -> bool:
        """Whether this forward is a candidate: decode steps and verify windows always, prefill chunks only
        with FT_KERNEL_PROFILE_PREFILL=1 and at least FT_KERNEL_PROFILE_MIN_ROWS rows."""
        if spec_verify or not is_prefill:
            return self.prefill != "only"
        return self.prefill in ("1", "only") and rows >= self.min_rows

    def due(self) -> bool:
        self.steps += 1
        return self.samples < self.limit and self.steps % self.every == 0

    @contextmanager
    def profile(self, label: str):
        from torch.profiler import ProfilerActivity, profile

        self.samples += 1
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            yield
            torch.cuda.synchronize()
        self.report(prof, label)

    def report(self, prof, label: str) -> None:
        rows = []
        for ev in prof.key_averages():
            # torch >= 2.5: device_time_total; older: cuda_time_total
            t = float(getattr(ev, "device_time_total", 0.0) or getattr(ev, "cuda_time_total", 0.0) or 0.0)
            if t <= 0.0 or ev.key.startswith(("ProfilerStep", "aten::", "cuda")):
                continue
            rows.append((t / 1e3, int(ev.count), ev.key))
        rows.sort(reverse=True)
        total = sum(r[0] for r in rows)
        buckets: dict[str, float] = {name: 0.0 for name, _ in _BUCKETS}
        buckets["other"] = 0.0
        for ms, _, key in rows:
            for name, rx in _BUCKETS:
                if rx.search(key):
                    buckets[name] += ms
                    break
            else:
                buckets["other"] += ms
        parts = " ".join(f"{k}={v:.1f}" for k, v in buckets.items() if v > 0.0)
        self.log(
            f"kernel profile [rank {self.rank}] {label}: GPU {total:.1f} ms in {sum(r[1] for r in rows)} kernels "
            f"({len(rows)} distinct): {parts}"
        )
        for ms, n, key in rows[:25]:
            self.log(f"kernel profile [rank {self.rank}]   {ms:7.2f} ms  n={n:5d}  {key[:90]}")
