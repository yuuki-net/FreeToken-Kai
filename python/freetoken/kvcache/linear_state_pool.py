from __future__ import annotations

import math

import torch
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.utils import div_even

_SSM_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def ssm_state_dtype() -> torch.dtype:
    """Recurrent (SSM) state dtype, from FREETOKEN_MAMBA_SSM_DTYPE (default fp32)."""
    return _SSM_DTYPES.get(str(ENV.MAMBA_SSM_DTYPE).lower(), torch.float32)


def _linear_local_dims(
    group: LinearGatedDeltaGroupConfig, tp_size: int
) -> tuple[int, int, int]:
    """TP-local ``(n_layers, conv_dim, v_heads)`` for the GDN state tensors -- the single
    source of the sharding math shared by the pool allocation and the byte estimate."""
    local_k_heads = div_even(group.num_key_heads, tp_size, allow_replicate=True)
    local_v_heads = div_even(group.num_value_heads, tp_size, allow_replicate=True)
    local_conv_dim = 2 * local_k_heads * group.key_head_dim + local_v_heads * group.value_head_dim
    return len(group.layer_ids), local_conv_dim, local_v_heads


class LinearStatePool:
    """Per-request recurrent state (conv + SSM) for GatedDeltaNet layers.

    Indexed by ``Req.table_idx`` (0..max_running_req), the same per-request slot the
    page table uses, so the scheduler's existing admit/free of ``table_idx`` covers the
    state's lifetime. One fixed slot per running request; no paging, no eviction.

    A model can declare extra per-request tensors on the same slots through
    ``ModelConfig.slot_states`` (see ``SlotStateSpec``); they advance, snapshot, COW and
    rebuild with the GDN state and are read back through ``slot_state(name, layer_id)``.
    Consumers must re-read them each forward: ``rebuild`` replaces the tensors.
    """

    def __init__(
        self,
        group: LinearGatedDeltaGroupConfig,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int | None = None,
        slot_states: tuple[SlotStateSpec, ...] = (),
    ) -> None:
        if tp_size is None:
            tp_size = get_tp_info().size

        self._group = group
        self._num_slots = num_slots
        self._device = device
        self._conv_dtype = dtype

        n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

        # conv left-context: the last (kernel-1) timesteps of the conv input stream.
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, group.conv_kernel_dim - 1),
            dtype=dtype,
            device=device,
        )
        # SSM recurrent state. fp32 by default (matches HF mamba_ssm_dtype); the dtype is
        # overridable via FREETOKEN_MAMBA_SSM_DTYPE (see ssm_state_dtype).
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, group.key_head_dim, group.value_head_dim),
            dtype=ssm_state_dtype(),
            device=device,
        )
        self._local_index = {layer_id: i for i, layer_id in enumerate(group.layer_ids)}

        self._slot_specs = tuple(slot_states)
        names = [spec.name for spec in self._slot_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate slot_state names: {names}")
        self._state_layer_index = {
            spec.name: {lid: i for i, lid in enumerate(spec.layer_ids)}
            for spec in self._slot_specs
        }
        self.slot_states: dict[str, torch.Tensor] = self._alloc_slot_states(num_slots)

        # Free-list allocator over slots 1..num_slots-1 (slot 0 reserved as a padding sink,
        # sglang MambaPool convention). Live working slots, ping-pong track slots, and
        # radix-tree-donated snapshots are all drawn from this single free-list, so memory
        # flows between them by demand. Unused by the op harness (which assigns slots by hand).
        self.padding_slot = 0
        self._free_slots: list[int] = list(range(1, num_slots))

        # Host tier (--linear-state-host-slots): a second, pinned home for snapshots the radix
        # tree holds but nobody is using. When the device slots run short, the least recently
        # used unlocked snapshot is copied down here and its device slot goes back to the
        # free-list, instead of the snapshot being thrown away. A host snapshot is never a live
        # state: it is only ever the source of a copy_from, which brings it back up. Host slot ids
        # live above HOST_BASE so a slot id alone says where the state is.
        self._host_slots = 0
        self._free_host: list[int] = []
        self._host_events: dict[int, torch.cuda.Event] = {}
        self.host_conv_states = self.host_recurrent_states = None
        self.host_slot_states: dict[str, torch.Tensor] = {}

    def _alloc_slot_states(self, num_slots: int, device=None, pin: bool = False) -> dict[str, torch.Tensor]:
        return {
            spec.name: torch.full(
                (max(1, len(spec.layer_ids)), num_slots, *spec.shape),
                spec.fill_value,
                dtype=spec.dtype if spec.dtype is not None else self._conv_dtype,
                device=self._device if device is None else device,
                pin_memory=pin,
            )
            for spec in self._slot_specs
        }

    # ---------------------------------------------------------------- host tier
    HOST_BASE = 1 << 30

    def enable_host_tier(self, host_slots: int) -> None:
        """Allocate ``host_slots`` pinned snapshot slots (0 = no host tier). Drops whatever
        the host tier held, so it is for startup and for an idle rebuild only."""
        self._host_slots = max(0, int(host_slots))
        self._free_host = list(range(self._host_slots))
        self._host_events = {}
        if self._host_slots == 0:
            self.host_conv_states = self.host_recurrent_states = None
            self.host_slot_states = {}
            return
        pin = self._device.type == "cuda"
        n = self._host_slots
        self.host_conv_states = torch.zeros(
            (self.conv_states.shape[0], n, *self.conv_states.shape[2:]),
            dtype=self.conv_states.dtype, pin_memory=pin,
        )
        self.host_recurrent_states = torch.zeros(
            (self.recurrent_states.shape[0], n, *self.recurrent_states.shape[2:]),
            dtype=self.recurrent_states.dtype, pin_memory=pin,
        )
        self.host_slot_states = self._alloc_slot_states(n, device="cpu", pin=pin)

    @property
    def host_slots(self) -> int:
        return self._host_slots

    @property
    def num_free_host_slots(self) -> int:
        return len(self._free_host)

    def is_host(self, slot: int) -> bool:
        return slot >= self.HOST_BASE

    def demote(self, slot: int) -> int:
        """Move the snapshot in device ``slot`` to a free host slot and give ``slot`` back to
        the device free-list. Returns the host slot id. The copy is queued on the current
        stream, which every later use of either slot is ordered behind: a forward waits for it
        before writing the freed device slot, and a copy_from back up queues after it."""
        assert not self.is_host(slot) and self._free_host, "demote needs a device slot and a free host slot"
        h = self._free_host.pop()
        self.host_conv_states[:, h].copy_(self.conv_states[:, slot], non_blocking=True)
        self.host_recurrent_states[:, h].copy_(self.recurrent_states[:, slot], non_blocking=True)
        for name, t in self.slot_states.items():
            self.host_slot_states[name][:, h].copy_(t[:, slot], non_blocking=True)
        if self._device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()
            self._host_events[h] = ev
        self._free_slots.append(slot)
        return self.HOST_BASE + h

    def state_views(self, slot: int) -> list[tuple[str, torch.Tensor]]:
        """``(name, [layers, *shape])`` views of one snapshot wherever it lives: conv,
        recurrent, then the declared slot states by name. A host view is waited for first, so
        the CPU can read it."""
        if self.is_host(slot):
            h = slot - self.HOST_BASE
            ev = self._host_events.get(h)
            if ev is not None:
                ev.synchronize()
            conv, rec, extra = self.host_conv_states, self.host_recurrent_states, self.host_slot_states
            i = h
        else:
            conv, rec, extra = self.conv_states, self.recurrent_states, self.slot_states
            i = slot
        return [("conv", conv[:, i]), ("recurrent", rec[:, i])] + [
            (f"slot.{n}", extra[n][:, i]) for n in sorted(extra)
        ]

    def has_slot_state(self, name: str) -> bool:
        return name in self.slot_states

    def slot_state(self, name: str, layer_id: int | None = None) -> torch.Tensor:
        """One declared sibling state, ``[num_slots, *shape]``; ``layer_id`` picks the layer row."""
        t = self.slot_states[name]
        if layer_id is None:
            assert not self._state_layer_index[name], (
                f"slot_state {name!r} is per-layer, pass layer_id"
            )
            return t[0]
        return t[self._state_layer_index[name][layer_id]]

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> list[int]:
        """Pop ``n`` free slot ids (LIFO). Raises if the pool is exhausted."""
        if n > len(self._free_slots):
            raise RuntimeError(
                f"LinearStatePool exhausted: need {n}, have {len(self._free_slots)}"
            )
        return [self._free_slots.pop() for _ in range(n)]

    def reclaim_all_slots(self) -> None:
        """Restore the free-list to all non-padding slots. Idle-only: the caller (e.g. a
        CacheManager rebuild that discards the tree owning donated snapshots) must guarantee no
        running request holds a slot, otherwise live state would be handed out twice. The host
        tier only ever holds tree snapshots, so it is emptied too."""
        self._free_slots = list(range(1, self._num_slots))
        self._free_host = list(range(self._host_slots))
        self._host_events = {}

    def rebuild(self, num_slots: int) -> None:
        """Reallocate the conv + recurrent state tensors for ``num_slots`` slots IN PLACE.

        Geometry (layers, conv dim, head dims) and dtypes are taken from the existing
        tensors; only the slot count changes. Object identity is preserved so cached
        references (ctx.linear_state_pool) stay valid. Idle-only and destructive: every
        live/snapshot state is dropped, so the caller must guarantee no running request
        holds a slot and the radix tree owning donated snapshots is discarded too.
        """
        n_layers, _, local_conv_dim, km1 = self.conv_states.shape
        _, _, local_v_heads, key_head_dim, value_head_dim = self.recurrent_states.shape
        conv_dtype, rec_dtype = self.conv_states.dtype, self.recurrent_states.dtype
        device = self._device
        self.conv_states = None
        self.recurrent_states = None
        self.slot_states = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self.conv_states = torch.zeros(
            (n_layers, num_slots, local_conv_dim, km1), dtype=conv_dtype, device=device
        )
        self.recurrent_states = torch.zeros(
            (n_layers, num_slots, local_v_heads, key_head_dim, value_head_dim),
            dtype=rec_dtype,
            device=device,
        )
        self.slot_states = self._alloc_slot_states(num_slots)
        self._num_slots = num_slots
        self._free_slots = list(range(1, num_slots))
        self._free_host = list(range(self._host_slots))  # the host tier's tensors keep their size
        self._host_events = {}

    def free(self, slots) -> None:
        """Return slot ids (device or host) to their free-list. Accepts an int, list, or 1-D tensor."""
        if isinstance(slots, torch.Tensor):
            slots = slots.flatten().tolist()
        elif isinstance(slots, int):
            slots = [slots]
        for s in slots:
            s = int(s)
            if s >= self.HOST_BASE:
                self._free_host.append(s - self.HOST_BASE)
            else:
                self._free_slots.append(s)

    def clear_slots(self, slots) -> None:
        """Zero conv + recurrent state at ``slots`` across all linear layers (fresh sequence)."""
        if isinstance(slots, (list, tuple)):
            slots = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self.conv_states[:, slots] = 0
        self.recurrent_states[:, slots] = 0
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, slots] = spec.fill_value

    def copy_from(self, src: int, dst: int) -> None:
        """Copy a whole-sequence snapshot (conv + recurrent, all layers) from slot ``src`` to
        ``dst``. Used for COW-on-restore (donated snapshot -> fresh live slot). ``src`` may be a
        host snapshot; ``dst`` is always a device slot."""
        if self.is_host(src):
            h = src - self.HOST_BASE
            self.conv_states[:, dst].copy_(self.host_conv_states[:, h], non_blocking=True)
            self.recurrent_states[:, dst].copy_(self.host_recurrent_states[:, h], non_blocking=True)
            for name, t in self.slot_states.items():
                t[:, dst].copy_(self.host_slot_states[name][:, h], non_blocking=True)
            return
        self.conv_states[:, dst].copy_(self.conv_states[:, src])
        self.recurrent_states[:, dst].copy_(self.recurrent_states[:, src])
        for t in self.slot_states.values():
            t[:, dst].copy_(t[:, src])

    def is_linear_layer(self, layer_id: int) -> bool:
        return layer_id in self._local_index

    def local_index(self, layer_id: int) -> int:
        return self._local_index[layer_id]

    def conv_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.conv_states[self._local_index[layer_id], table_idx]

    def recurrent_state(self, layer_id: int, table_idx: int) -> torch.Tensor:
        return self.recurrent_states[self._local_index[layer_id], table_idx]

    def reset(self, table_idx: int) -> None:
        """Zero a slot across all linear layers (new request takes this table_idx)."""
        self.conv_states[:, table_idx].zero_()
        self.recurrent_states[:, table_idx].zero_()
        for spec in self._slot_specs:
            self.slot_states[spec.name][:, table_idx] = spec.fill_value

    @property
    def num_linear_layers(self) -> int:
        return len(self._local_index)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def device(self) -> torch.device:
        return self._device

    def bytes_per_slot(self) -> int:
        """Total state bytes for one request (all linear layers)."""
        per = (
            self.conv_states[:, 0].numel() * self.conv_states.element_size()
            + self.recurrent_states[:, 0].numel() * self.recurrent_states.element_size()
        )
        for t in self.slot_states.values():
            per += t[:, 0].numel() * t.element_size()
        return int(per)


def linear_state_bytes_per_req(
    group: LinearGatedDeltaGroupConfig,
    tp_size: int,
    dtype: torch.dtype,
    slot_states: tuple[SlotStateSpec, ...] = (),
) -> int:
    """Linear-state bytes for one request across all linear layers (TP-local), plus any
    declared slot_states."""
    n_layers, local_conv_dim, local_v_heads = _linear_local_dims(group, tp_size)

    conv_elems = local_conv_dim * (group.conv_kernel_dim - 1)
    rec_elems = local_v_heads * group.key_head_dim * group.value_head_dim
    conv_bytes = conv_elems * dtype.itemsize  # conv state in model dtype
    rec_bytes = rec_elems * ssm_state_dtype().itemsize  # recurrent state (default fp32)
    total = n_layers * (conv_bytes + rec_bytes)

    for spec in slot_states:
        item = (spec.dtype if spec.dtype is not None else dtype).itemsize
        total += max(1, len(spec.layer_ids)) * math.prod(spec.shape) * item
    return int(total)


__all__ = ["LinearStatePool", "linear_state_bytes_per_req"]


def state_pool_bytes(config, num_slots: int | None = None) -> int:
    """Total GDN state-pool bytes at ``num_slots`` PHYSICAL slots (default: the startup
    slot count). The engine adds this to the KV family's fixed cost when budgeting --
    the state pool is a sibling pool, not a KV tier."""
    linear_group = config.model_config.linear_attention_group()
    slot_states = getattr(config.model_config, "slot_states", ())
    if linear_group is None:
        if slot_states:
            raise ValueError("slot_states ride the linear-state slots; model has no linear group")
        return 0
    slots = num_slots if num_slots is not None else _linear_pool_num_slots(config)
    per_req = linear_state_bytes_per_req(
        linear_group, int(getattr(config, "tp_size", None) or config.tp_info.size), config.dtype, slot_states
    )
    return per_req * slots


def _linear_pool_num_slots(config) -> int:
    """LinearStatePool slot count. Hybrid-radix non-evictable peak is 4 slots per running request
    (1 live + 2 ping-pong + 1 committed snapshot locked through decode), plus a cross-request
    snapshot cache and a padding sink; naive GDN keeps the old (max_running_req + 1)."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1  # live + dummy/padding
    ratio = config.linear_state_cache_ratio
    n_cache = max(4, int(ratio * mr))
    return 4 * mr + n_cache + 1  # live + 2 ping-pong + locked committed snapshot + cache + padding


def _linear_pool_min_slots(config) -> int:
    """Floor on LinearStatePool slots that still runs: the non-evictable working set with a
    zero snapshot cache. Hybrid-radix needs 4 per running request (1 live + 2 ping-pong + 1
    committed snapshot locked through decode) + the padding sink; naive needs 1 per request +
    padding. Below this, a full max_running_req batch can't get its slots and admission
    deadlocks -- so a runtime rebuild rejects a smaller request."""
    mr = config.max_running_req
    if config.cache_type != "hybrid_radix":
        return mr + 1
    return 4 * mr + 1
