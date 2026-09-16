"""A launch profile the console can propose for THIS host and THIS model.

Not a tuner: it reads what the machine has (GPUs, VRAM, RAM, cores, the model's own config) and
writes down the settings this fork's own measurements point to, each with the reason it is there.
Nothing is applied — the editor shows the flags and the person decides.

stdlib + the torch-free model reader only: the manager imports this."""

from __future__ import annotations

import json
import math
import os
import subprocess

from .models import resolve_model

GiB = 1 << 30


# ------------------------------------------------------------------ the host
def gpus() -> list[dict]:
    """[{index, name, total_bytes, free_bytes}] from nvidia-smi; [] when there is no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            found.append({"index": int(parts[0]), "name": parts[1],
                          "total_bytes": int(float(parts[2])) * (1 << 20), "free_bytes": int(float(parts[3])) * (1 << 20),
                          "compute_cap": float(parts[4]) if len(parts) > 4 and parts[4] else None})
        except ValueError:
            continue
    return found


def host_memory() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                key = {"MemTotal": "total", "MemAvailable": "available"}.get(k)
                if key:
                    out[key] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


# bytes per stored parameter, scales included: a 4-bit checkpoint is not 0.5 bytes/param on disk
BYTES_PER_PARAM = {"nvfp4": 0.58, "modelopt": 0.58, "mxfp4": 0.53, "awq": 0.6, "gptq": 0.6, "fp8": 1.06, None: 2.0}


def physical_cores() -> int:
    """Cores, not threads: the CPU expert workers get no benefit from the second thread of a core."""
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
    except OSError:
        return os.cpu_count() or 4
    pairs = set()
    phys = core = None
    for line in text.splitlines():
        if line.startswith("physical id"):
            phys = line.split(":")[1].strip()
        elif line.startswith("core id"):
            core = line.split(":")[1].strip()
            pairs.add((phys, core))
    return len(pairs) or (os.cpu_count() or 4)


# ------------------------------------------------------------------ the model
def model_facts(path: str) -> dict:
    """What the checkpoint says about itself, plus the size of its weights on disk."""
    facts: dict = {"weight_bytes": 0}
    if os.path.isdir(path):
        for name in os.listdir(path):
            if name.endswith((".safetensors", ".gguf", ".bin")):
                try:
                    facts["weight_bytes"] += os.stat(os.path.join(path, name)).st_size
                except OSError:
                    pass
    try:
        with open(os.path.join(path, "config.json")) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return facts
    text = cfg.get("text_config") or {}
    get = lambda key, default=None: cfg.get(key, text.get(key, default))  # noqa: E731
    quant = cfg.get("quantization_config") or text.get("quantization_config") or {}
    layer_types = get("layer_types") or []
    facts.update({
        "layer_types": layer_types,
        "hidden_size": get("hidden_size"),
        "moe_intermediate_size": get("moe_intermediate_size") or get("intermediate_size"),
        "model_type": cfg.get("model_type"),
        "architecture": (cfg.get("architectures") or [None])[0],
        "num_layers": get("num_hidden_layers"),
        "num_experts": get("num_experts") or get("n_routed_experts") or get("num_local_experts"),
        "num_kv_heads": get("num_key_value_heads"),
        "head_dim": get("head_dim") or (get("hidden_size") and get("num_attention_heads")
                                        and get("hidden_size") // get("num_attention_heads")),
        "max_context": get("max_position_embeddings"),
        "quant": quant.get("quant_method") or quant.get("quant_algo"),
        "vision": bool(cfg.get("vision_config") or text.get("vision_config")),
    })
    return facts


def kv_layers(facts: dict) -> int:
    """Layers that hold a KV cache. A GDN hybrid keeps one every few layers (the rest carry a
    recurrent state instead), so counting every layer overstates the KV cost several times over."""
    types = facts.get("layer_types") or []
    if types and any("linear" in str(x) for x in types):
        return sum(1 for x in types if "full" in str(x)) or len(types)
    return facts.get("num_layers") or 0


def kv_bytes_per_token(facts: dict, quantized: bool) -> int:
    """K+V for one token, 16-bit unless the KV cache is q8_0 (1.88x smaller)."""
    layers, kv_heads, head_dim = kv_layers(facts), facts.get("num_kv_heads"), facts.get("head_dim")
    if not (layers and kv_heads and head_dim):
        return 0
    per_token = 2 * layers * kv_heads * head_dim * 2
    return int(per_token / 1.88) if quantized else per_token


def expert_bytes(facts: dict) -> int:
    """Roughly what the routed experts weigh: they are read per token, not resident, so the VRAM
    left for KV is the weights MINUS this."""
    experts, layers = facts.get("num_experts"), facts.get("num_layers")
    hidden, inter = facts.get("hidden_size"), facts.get("moe_intermediate_size")
    if not (experts and layers and hidden and inter):
        return 0
    bpp = BYTES_PER_PARAM.get(facts.get("quant"), 2.0)
    return int(layers * experts * 3 * hidden * inter * bpp)  # gate, up, down


# ------------------------------------------------------------------ the recommendation
def recommend(model: str, *, extra_dirs: list[str] | None = None) -> dict:
    """{flags: [...], notes: [{flag, why}], host: {...}} — the flags a person would otherwise
    have to read the guides for. Each note says why, so nothing has to be taken on faith."""
    path = resolve_model(model, extra_dirs)
    facts = model_facts(path) if os.path.isdir(path) else {}
    cards, mem, cores = gpus(), host_memory(), physical_cores()
    vram = min((g["total_bytes"] for g in cards), default=0)
    total_vram = sum(g["total_bytes"] for g in cards)
    weights = facts.get("weight_bytes", 0)
    is_moe = bool(facts.get("num_experts"))
    experts_w = expert_bytes(facts) if is_moe else 0
    # what has to sit on the GPU: everything except the routed experts, which stream in per token
    resident = max(int(weights * 0.1), weights - experts_w) if weights else 0

    flags: list[str] = []
    notes: list[dict] = []

    def add(flag: str, value: str | None, why: str, why_en: str) -> None:
        flags.append(flag)
        if value is not None:
            flags.append(value)
        notes.append({"flag": flag, "value": value, "why": why, "why_en": why_en})

    # --- bf16 is emulated before Ampere: measured 592 -> 285 tok/s of prompt processing on a 2060
    caps = [g.get("compute_cap") for g in cards if g.get("compute_cap")]
    if caps and max(caps) < 8.0:
        sm = int(min(caps) * 10)
        add("--dtype", "float16",
            f"この GPU（sm_{sm}）は bfloat16 を持たないので、float16 で動かします。"
            "指定しないとプロンプト処理が半分程度まで落ちます（2060・Ornith で 592 → 285 tok/s の実測）。"
            "ただし読み込み時に変換のコピーが増えるので、VRAM がぎりぎりのモデルでは起動できないことがあります（その場合はこの行を外します）。",
            f"This GPU (sm_{sm}) has no bfloat16, so run in float16. Without it prompt processing drops to about half "
            "(592 -> 285 tok/s measured on a 2060 with Ornith). The conversion needs an extra copy while loading, so a model "
            "that barely fits in VRAM may not start with it (drop this line then).")

    # --- how many cards, and does the model fit on one?
    if len(cards) > 1 and resident and resident > vram * 0.75:
        add("--pp-size", str(len(cards)),
            f"エキスパート以外の重みが約 {resident / GiB:.1f} GiB で 1 枚（{vram / GiB:.0f} GiB）に収まらないので、層を {len(cards)} 枚に分けます。",
            f"The weights other than the experts are about {resident / GiB:.1f} GiB and do not fit one {vram / GiB:.0f} GiB card, "
            f"so the layers are split across {len(cards)}.")
        add("--gpu", ",".join(str(g["index"]) for g in cards), "pp のランク順に GPU を並べます。", "The GPUs in pipeline rank order.")
    elif len(cards) > 1:
        add("--gpu", "0", "モデルは 1 枚に収まるので、GPU 0 だけを使います。", "The model fits one card, so only GPU 0 is used.")

    # --- experts: where they run, and how much RAM they may take
    if is_moe:
        if vram <= 8 * GiB:
            add("--moe-strategy", "hybrid",
                f"VRAM が {vram / GiB:.0f} GiB と小さいので、GPU に載らないエキスパートは CPU で計算します。",
                f"With {vram / GiB:.0f} GiB of VRAM, experts that are not on the GPU are computed on the CPU.")
            add("--moe-cpu-layers", "auto",
                "CPU に回す層を実測から決めます（WSL では GPU に固定できる RAM に上限があります）。",
                "Which layers run on the CPU is decided from measurements (under WSL the RAM that can be pinned for the GPU is capped).")
            add("--moe-cpu-threads", str(max(2, min(cores - 2, 8))),
                f"物理コア {cores} 個から、ほかの処理のぶんを残した数です。",
                f"{cores} physical cores, leaving some for everything else.")
        add("--moe-cache-auto", None,
            "空いている VRAM から、GPU に載せるエキスパートの枠を自動で決めます。",
            "The number of experts kept on the GPU is sized from the free VRAM.")
        if weights and mem.get("total") and weights > mem["total"]:
            cap = max(8, int((mem["total"] * 0.7) / GiB))
            add("--moe-bank-ram", f"{cap}G",
                f"重み {weights / GiB:.0f} GiB が RAM {mem['total'] / GiB:.0f} GiB に収まらないので、入る分だけ RAM に置き、残りは SSD から読みます。",
                f"{weights / GiB:.0f} GiB of weights do not fit {mem['total'] / GiB:.0f} GiB of RAM: keep what fits in RAM and read the rest from the SSD.")
        if vram - resident < 2 * GiB and vram <= 8 * GiB:
            add("--disable-moe-prefill-overlap", None,
                "プロンプト処理の 2 バッファ分の VRAM が残らないので、重ね合わせを切ります。",
                "There is no VRAM left for prompt processing's second buffer, so the overlap is turned off.")

    # --- context: what fits in what is left after the weights
    # Context: a value that starts, not a prediction. What actually fits depends on buffers this
    # cannot see, so take the card's size as the tier and say where to look to raise it.
    max_ctx = facts.get("max_context") or 0
    per_token = kv_bytes_per_token(facts, quantized=is_moe)
    if max_ctx:
        tier = 16384 if vram <= 8 * GiB else 65536 if vram <= 16 * GiB else 131072
        ctx = min(max_ctx, tier)
        if is_moe:
            add("--kv-cache-dtype", "q8_0", "KV を量子化して、空いた VRAM をエキスパートに回します。",
                "Quantize the KV cache and give the VRAM it frees to the experts.")
        size = f"{ctx * per_token / GiB:.1f} GiB" if per_token else None
        add("--kv-reserve-tokens", str(ctx),
            f"まず起動できる長さです（KV {size or '不明'}、VRAM {vram / GiB:.0f} GiB のカード向け）。起動後に「改善提案」が GPU の空きを見て伸ばす値を出します。",
            f"A length that starts (KV {size or 'unknown'}, for a {vram / GiB:.0f} GiB card). Once the server runs, "
            "“From measurements” proposes a longer one from the free VRAM.")
        add("--max-seq-len-override", str(ctx), "宣伝する長さと実際に入る長さをそろえます。",
            "The advertised context matches what actually fits.")
    if facts.get("model_type") in ("qwen3_5_moe", "qwen4_exp") and vram <= 8 * GiB:
        add("--host-embedding", None, "埋め込み表を RAM に置いて、その分の VRAM を KV に回します。",
            "Keep the embedding table in RAM and give its VRAM to the KV cache.")

    # --- the rest
    add("--memory-ratio", "0.85", "画面表示やほかのアプリが使う VRAM の余地を残します。",
        "Leave VRAM for the display and other applications.")
    add("--max-running-req", "1", "1 リクエストずつ処理します（小さいカードでは同時実行より安定します）。",
        "One request at a time (steadier than concurrency on small cards).")
    if facts.get("vision"):
        add("--mm-encoder-weights", "cpu", "画像の処理を CPU で行い、VRAM を使いません。",
            "Images are encoded on the CPU, using no VRAM.")
    add("--host", "0.0.0.0", "同じ LAN のほかの PC や Docker からつながるようにします。",
        "Reachable from other PCs on the LAN and from Docker.")
    add("--decode-log-interval", "5", "生成中の速度をログで追えるようにします。",
        "Generation speed shows up in the log while it runs.")

    return {
        "model": path,
        "flags": flags,
        "notes": notes,
        "host": {
            "gpus": cards, "cores": cores, "memory": mem, "resident_bytes": resident, "expert_bytes": experts_w,
            "model": {k: v for k, v in facts.items() if k != "weight_bytes"},
            "weight_bytes": weights,
        },
    }
