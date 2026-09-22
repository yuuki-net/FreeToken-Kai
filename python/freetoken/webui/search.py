"""What the benchmark tries on the running model, and what it deliberately leaves alone.

Every ``ft serve`` flag that can change speed is either a candidate here or in ``SKIPPED`` with the
reason it is not measured. A candidate is one change against the best settings found so far; the
tuner keeps it only when the measurement says so (``keep``). Which candidates apply depends on the
model (MoE, GDN, MTP head, expert format), the host (GPUs, their links, RAM) and the base flags.

"standard" runs the candidates that moved speed in this fork's measurements (an RTX 2060, two RTX 3060s);
"thorough" adds the ones that have not, or only rarely, so a machine unlike ours gets the chance.

stdlib only: the manager imports this."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

# the expert formats whose kernels --quant-backend can pick, as ft serve names them
MOE_KERNELS = {"nvfp4": ("triton", "marlin", "b12x"), "mxfp4": ("triton", "triton_gptoss")}
LINEAR_KERNELS = {"nvfp4": ("marlin", "triton")}


@dataclass
class Candidate:
    key: str
    changes: dict
    drop: tuple = ()
    # which figure has to improve: "gen" (generation, weighted by the chosen use), "prefill", or
    # "either" (one of them by its gain while the other holds)
    goal: str = "gen"
    gain: float = 1.03
    hold_gen: float = 0.97
    hold_prefill: float = 0.90
    thorough: bool = False
    after: str | None = None  # only when this earlier candidate was kept
    what_ja: str = ""
    what_en: str = ""
    code: bool = False  # generation on code is measured too (MTP acceptance depends on the text)
    # candidates that undo each other share a group: once one is tried, the others are not
    group: str = ""

    def __post_init__(self) -> None:
        self.group = self.group or self.key

    def change(self) -> dict:
        if self.changes:
            flag, value = next(iter(self.changes.items()))
            return {"flag": flag, "value": value}
        return {"flag": self.drop[0], "value": None, "removed": True}


def flag_value(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def merge_changes(args: list[str], changes: dict) -> dict:
    """--quant-backend holds one entry per kernel table: a new entry joins the kept ones."""
    if "--quant-backend" not in changes:
        return changes
    entries = {}
    for item in (flag_value(args, "--quant-backend") or "").split(",") + [changes["--quant-backend"] or ""]:
        if "=" in item:
            table, name = item.split("=", 1)
            entries[table.strip()] = name.strip()
    return dict(changes, **{"--quant-backend": ",".join(f"{t}={n}" for t, n in sorted(entries.items()))})


def expert_format(quant: str | None) -> str | None:
    q = (quant or "").lower()
    if q in ("modelopt", "nvfp4", "modelopt_fp4"):
        return "nvfp4"
    if q == "mxfp4":
        return "mxfp4"
    return None


def installed_modules() -> set[str]:
    """The optional packages some kernels need; ft serve runs in the same environment as ft mgr."""
    out = set()
    for name in ("vllm", "flashinfer"):
        try:
            if importlib.util.find_spec(name) is not None:
                out.add(name)
        except (ImportError, ValueError):
            pass
    return out


def kernel_unusable(name: str, hw: dict, modules: set[str]) -> tuple[str, str] | None:
    """Why a kernel cannot run on this PC (the same checks the engine makes when it loads), or None."""
    caps = [g.get("compute_cap") for g in ((hw or {}).get("measurements") or {}).get("gpus") or [] if g.get("compute_cap")]
    if name == "marlin" and "vllm" not in modules:
        return "vLLM が入っていないので使えない。", "vLLM is not installed."
    if name == "b12x":
        if caps and min(caps) < 12.0:
            return f"sm_120 以上の GPU が必要（この GPU は sm_{int(min(caps) * 10)}）。", f"Needs an sm_120 GPU or newer (this one is sm_{int(min(caps) * 10)})."
        if "flashinfer" not in modules:
            return "flashinfer が入っていないので使えない。", "flashinfer is not installed."
    return None


def offload_unpinnable(hw: dict) -> tuple[str, str] | None:
    """Why a plain ``offload`` start cannot happen on this host, when it cannot.

    ``offload`` page-locks every expert bank, so a host whose measured cap is short of them
    refuses the start; the candidate is then not worth a trial (FreeToken-Kai#2). The figures
    come from the hardware measurement, which measures the cap before anything else pins."""
    pin = ((hw or {}).get("measurements") or {}).get("pin") or {}
    cap, banks = pin.get("cap_bytes"), pin.get("banks_bytes")
    if not cap or not banks or banks <= cap:
        return None
    gib = 1 << 30
    return (f"このマシンがページロックできる RAM は実測 {cap / gib:.2f} GiB で、エキスパート {banks / gib:.1f} GiB を"
            "全部固定する offload は起動しないので試さない。",
            f"This machine page-locks {cap / gib:.2f} GiB (measured), so a plain offload start, which pins all "
            f"{banks / gib:.1f} GiB of experts, does not start and is not tried.")


def unavailable(facts: dict, hw: dict, modules: set[str] | None = None) -> list[dict]:
    """What was left out of the plan, for the page's list of what was not measured."""
    modules = installed_modules() if modules is None else modules
    fmt = expert_format(facts.get("quant")) if facts.get("num_experts") else None
    out = []
    if facts.get("num_experts"):
        why = offload_unpinnable(hw)
        if why:
            out.append({"flag": "--moe-strategy offload", "why": why[0], "why_en": why[1]})
    for kind, table in (("moe", MOE_KERNELS), ("linear", LINEAR_KERNELS)):
        for name in table.get(fmt or "", ()):
            why = kernel_unusable(name, hw, modules)
            if why:
                out.append({"flag": f"--quant-backend {kind}.{fmt}={name}", "why": why[0], "why_en": why[1]})
    return out


def plan(args: list[str], facts: dict, hw: dict, mode: str = "standard", modules: set[str] | None = None) -> list[Candidate]:
    modules = installed_modules() if modules is None else modules
    m = (hw or {}).get("measurements") or {}
    is_moe = bool(facts.get("num_experts"))
    model_type = facts.get("model_type")
    gdn = any("linear" in str(x) for x in facts.get("layer_types") or [])
    pp = int(flag_value(args, "--pp-size") or 1)
    strategy = flag_value(args, "--moe-strategy")
    out: list[Candidate] = []

    def add(c: Candidate) -> None:
        if mode == "thorough" or not c.thorough:
            out.append(c)

    if is_moe:
        # --- how the experts run: the kernel bench cannot see the whole step, so both are started
        if strategy == "hybrid" and not offload_unpinnable(hw):
            add(Candidate("strategy_offload", {"--moe-strategy": "offload"}, drop=("--moe-cpu-layers", "--moe-cpu-threads"), group="strategy",
                          what_ja="足りないエキスパートを CPU で計算せず、すべて GPU に送る（offload）",
                          what_en="send every missing expert to the GPU instead of computing some on the CPU (offload)"))
        elif strategy in (None, "offload", "auto"):
            add(Candidate("strategy_hybrid", {"--moe-strategy": "hybrid", "--moe-cpu-layers": "auto"}, group="strategy",
                          what_ja="足りないエキスパートの一部を CPU で計算する（hybrid）",
                          what_en="compute some of the missing experts on the CPU (hybrid)"))
        cpu = m.get("cpu_moe") or {}
        if strategy == "hybrid" and cpu.get("cores") and str(cpu["cores"]) != flag_value(args, "--moe-cpu-threads"):
            add(Candidate("threads_all", {"--moe-cpu-threads": str(cpu["cores"])},
                          what_ja=f"CPU スレッドを全物理コア（{cpu['cores']}）にする。カーネル単体では頭打ちでも、1 ステップ全体では違うことがある",
                          what_en=f"use every physical core ({cpu['cores']}) for the CPU experts; the whole step can differ from the kernel alone"))
        threads = flag_value(args, "--moe-cpu-threads")
        if strategy == "hybrid" and threads and threads.isdigit() and int(threads) > 2:
            fewer = str(int(threads) - 1)
            add(Candidate("threads_fewer", {"--moe-cpu-threads": fewer},
                          what_ja=f"CPU スレッドを 1 つ減らして {fewer} にする（空いたコアで GPU への指示や転送が待たされなくなることがある）",
                          what_en=f"one CPU thread fewer ({fewer}): the core left free can keep GPU launches and transfers from waiting"))
        if strategy == "hybrid":
            add(Candidate("fetch_none", {"--moe-hybrid-max-fetch": "0"}, thorough=True,
                          what_ja="足りないエキスパートを GPU へ送らず、すべて CPU で計算する",
                          what_en="fetch no missing expert over PCIe: the CPU computes them all"))

        # --- kernels for the expert format
        fmt = expert_format(facts.get("quant"))
        for name in MOE_KERNELS.get(fmt or "", ()):
            if kernel_unusable(name, hw, modules):
                continue
            add(Candidate(f"kernel_moe_{name}", {"--quant-backend": f"moe.{fmt}={name}"}, goal="either",
                          what_ja=f"エキスパートの {fmt} カーネルを {name} にする", what_en=f"run the {fmt} experts with the {name} kernel"))
        for name in LINEAR_KERNELS.get(fmt or "", ()):
            if kernel_unusable(name, hw, modules):
                continue
            add(Candidate(f"kernel_linear_{name}", {"--quant-backend": f"linear.{fmt}={name}"}, goal="either", thorough=True,
                          what_ja=f"エキスパート以外の {fmt} 層のカーネルを {name} にする（その形式の層があるときだけ差が出る）",
                          what_en=f"run the non-expert {fmt} layers with the {name} kernel (only matters when there are such layers)"))

        # --- the KV cache: VRAM it frees goes to the expert cache, its decode costs time
        kv = flag_value(args, "--kv-cache-dtype")
        if kv in ("q8_0", "q4_0"):
            add(Candidate("kv_16bit", {}, drop=("--kv-cache-dtype",), group="kv",
                          what_ja="KV を量子化しない（16bit）。KV の読み出しは速いが、エキスパートの枠は減る",
                          what_en="keep the KV cache at 16 bits: faster to read, fewer expert slots"))
        else:
            add(Candidate("kv_q8", {"--kv-cache-dtype": "q8_0"}, group="kv",
                          what_ja="KV を q8_0 にして、空いた VRAM をエキスパートの枠に回す",
                          what_en="store the KV cache as q8_0 and give the VRAM to the expert cache"))
        if model_type != "gpt_oss" and kv != "q4_0":
            add(Candidate("kv_q4", {"--kv-cache-dtype": "q4_0"}, thorough=True,
                          what_ja="KV を q4_0 にする（さらに小さいが、読み出しの展開が重い）",
                          what_en="store the KV cache as q4_0 (smaller still, heavier to decode)"))

        # --- prefill: the overlap of two expert buffers, and resident rows copied on the GPU
        if flag_value(args, "--disable-moe-prefill-overlap") is not None:
            add(Candidate("overlap_on", {}, drop=("--disable-moe-prefill-overlap",), goal="prefill", gain=1.05,
                          what_ja="プロンプト処理の 2 バッファの重ね合わせを有効にする（VRAM が足りれば速くなる）",
                          what_en="enable the two-buffer overlap for prefill expert copies (faster when the VRAM allows)"))
        add(Candidate("hit_d2d", {"--moe-prefill-hit-d2d": None}, goal="prefill", gain=1.05, thorough=True,
                      what_ja="プロンプト処理で GPU に載っているエキスパートは GPU 内でコピーし、足りない分だけ送る",
                      what_en="copy the experts already on the GPU device-side during prefill and stream only the misses"))

        if flag_value(args, "--moe-bank-ram") is not None:
            add(Candidate("bank_prefetch", {"--moe-bank-prefetch": None},
                          what_ja="CPU が計算する前に、SSD 側のエキスパートの行を先読みさせる",
                          what_en="ask the kernel for the SSD-side expert rows before the CPU computes a layer"))

    # --- prefill width
    if flag_value(args, "--prefill-chunk-budget") is None:
        add(Candidate("budget_075", {"--prefill-chunk-budget": "0.75"}, goal="prefill", gain=1.05,
                      what_ja="プロンプト処理の 1 チャンクが空き VRAM の 75% まで使えるようにして、チャンクを広げる",
                      what_en="let one prefill chunk take 75% of the free VRAM, so chunks are wider"))
    add(Candidate("budget_090", {"--prefill-chunk-budget": "0.9"}, goal="prefill", gain=1.05, thorough=True, after="budget_075",
                  what_ja="チャンクに空き VRAM の 90% まで使わせる", what_en="let one prefill chunk take 90% of the free VRAM"))
    pieces = flag_value(args, "--prefill-mixer-pieces")
    if gdn and (model_type == "qwen4_exp" or (model_type == "qwen3_5_moe" and pp == 1)):
        if pieces != "4":
            add(Candidate("pieces_4", {"--prefill-mixer-pieces": "4", "--max-prefill-length": "16384"}, goal="prefill", gain=1.05, group="pieces",
                          what_ja="GDN と attention を 4 つに分け、チャンクの上限を 16384 に上げる（2060・Ornith で 722 tok/s の実測）",
                          what_en="run GDN and attention in four pieces with a 16384-token chunk ceiling (722 tok/s on a 2060 with Ornith)"))
        if pieces:
            add(Candidate("pieces_1", {}, drop=("--prefill-mixer-pieces",), goal="prefill", gain=1.05, thorough=True, group="pieces",
                          what_ja="GDN と attention を分けない（片分けが逆効果になる機械の確認）",
                          what_en="do not split GDN and attention (for a machine where the pieces cost more than they give)"))
    add(Candidate("max_prefill_16k", {"--max-prefill-length": "16384"}, goal="prefill", gain=1.05, thorough=True,
                  what_ja="チャンクの上限を 16384 に上げる", what_en="raise the prefill chunk ceiling to 16384 tokens"))

    # --- memory placement
    if flag_value(args, "--host-embedding") is not None:
        add(Candidate("no_host_embedding", {}, drop=("--host-embedding",), goal="either", thorough=True,
                      what_ja="埋め込み表を GPU に置く（VRAM を使うが、読み出しは GPU 内で済む）",
                      what_en="keep the embedding table on the GPU (uses VRAM, no PCIe gather)"))
    if model_type == "qwen4_exp":
        if flag_value(args, "--dense-quant") == "fp8":
            add(Candidate("dense_bf16", {}, drop=("--dense-quant",), goal="either", thorough=True,
                          what_ja="エキスパート以外の重みを fp8 にせず bf16 のままにする",
                          what_en="keep the non-expert weights at bf16"))
        ram, weights = (facts.get("ram_total") or 0), (facts.get("weight_bytes") or 0)
        if ram and weights and ram > weights * 1.15:
            add(Candidate("ple_pinned", {"--ple-backend": "pinned"},
                          what_ja="PLE の表を RAM に常駐させる（RAM に余裕があるときだけ）",
                          what_en="keep the PLE table in page-locked RAM (only when RAM has room)"))

    # --- two GPUs: where the split falls, and how the ranks hand chunks over
    if pp > 1:
        layers = facts.get("num_layers") or 0
        links = [((m.get("pcie") or {}).get(g) or {}).get("h2d_gbs") for g in (flag_value(args, "--gpu") or "").split(",") if g]
        if pp == 2 and layers and len(links) == 2 and all(links):
            # give the rank on the faster link more layers: its experts cross PCIe faster
            toward = 1 if links[0] > links[1] * 1.2 else -1 if links[1] > links[0] * 1.2 else 0
            if toward:
                for k, thorough in ((1, False), (2, True)):
                    split = layers // 2 + toward * k
                    add(Candidate(f"pp_layers_{split}", {"--pp-layers": str(split)}, goal="either", thorough=thorough,
                                  what_ja=f"層の分け方を {split} / {layers - split} にして、速いリンクの GPU に多く載せる",
                                  what_en=f"split the layers {split} / {layers - split}, more on the GPU with the faster link"))
        add(Candidate("send_ahead_3", {"--pp-send-ahead": "3"}, goal="prefill", gain=1.05, thorough=True,
                      what_ja="1 枚目が 2 枚目へ残差を 3 チャンク先まで送れるようにする",
                      what_en="let the first rank have three residual streams in flight to the next"))
        if model_type == "qwen4_exp":
            add(Candidate("prefill_group_2", {"--pp-prefill-group": "2"}, goal="prefill", gain=1.05, thorough=True,
                          what_ja="2 枚目がチャンクを 2 つためて層ごとにまとめて処理する",
                          what_en="let the second rank hold two chunks and run them layer by layer"))

    # --- speculative decoding with the checkpoint's MTP head: its gain depends on the text
    if facts.get("mtp") and flag_value(args, "--max-running-req") in ("1", None):
        for k in ("3", "5"):
            add(Candidate(f"mtp_{k}", {"--spec-mtp": k}, code=True, hold_prefill=0.85,
                          what_ja=f"MTP で {k} トークン先まで予測して、まとめて確かめる（コードやツール呼び出しで効きやすい）",
                          what_en=f"draft {k} tokens ahead with the MTP head and verify them together (helps most on code and tool calls)"))
    return out


# flags that can change speed but are not searched, with the reason shown on the page
SKIPPED = [
    ("--memory-ratio", "上げると速くなりうるが、あとで画面やほかのアプリが VRAM を使ったときに落ちる。その危険は測定では見えないので、0.85 のままにする。",
     "Raising it can help, but the server dies once the display or another app takes VRAM later; a benchmark cannot see that risk, so 0.85 stays."),
    ("--max-running-req", "複数の要求を同時に処理する速さ（スループット）で、1 人で使う速さではないので 1 のまま。",
     "It trades single-request speed for concurrent throughput; one person's speed is measured, so it stays 1."),
    ("--cuda-graph-max-bs", "同時処理が 1 のときは使われない。", "Unused with one running request."),
    ("--moe-cpu-layers", "WSL で GPU に固定できる RAM の上限（このベンチマークが実測して記録する）から auto が層を決める。上限を超える指定は起動できないので探さない。",
     "Under WSL auto picks the layers from the pinnable-RAM limit, which this benchmark measures and records; a list beyond it does not start, so it is not searched."),
    ("--attention-backend", "GPU とモデルに合うものを auto が選ぶ。合わないものは起動しないか結果が崩れるので探さない。",
     "auto picks the backend that fits the GPU and model; others fail to start or change the output."),
    ("--moe-bank-ram", "RAM に重みが入らないときに必要かどうかで決まる（速さの比較ではない）。",
     "Decided by whether the weights fit RAM, not by speed."),
    ("--linear-state-cache-ratio / --prefix-disk-cache / --enable-special-token-ckpt",
     "会話の続きや切り替えのときの待ち時間に効く。1 回のプロンプトの測定では差が出ないので、今は測っていない。",
     "They shorten follow-up and switched conversations; a single-prompt measurement cannot show them yet."),
    ("--moe-bank-rewarm", "アイドル中に追い出されたページを読み戻す機能で、待ち時間の測定には何分もの放置が要る。",
     "It reads evicted pages back while idle; measuring it takes minutes of idling."),
    ("--mm-encoder-weights / --image-*", "画像入力の速さと VRAM に効く。文章の測定では差が出ない。", "They affect image input, not text."),
    ("--expert-load", "起動にかかる時間だけに効く。", "Only affects load time."),
    ("--dtype", "Turing では bfloat16 が使えないので float16 に決まる（起動できなければ外す）。それ以外の GPU では auto。",
     "Turing has no bfloat16, so float16 (dropped if the model does not load); auto elsewhere."),
]
