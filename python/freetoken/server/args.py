from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
from freetoken.distributed import DistributedInfo
from freetoken.scheduler import SchedulerConfig
from freetoken.utils import init_logger

logger = init_logger(__name__)


class _DeprecatedAlias(argparse.Action):
    """An old flag: warns at parse time, converts the value if asked, stores it."""

    def __init__(self, *args, new_flag: str, convert=None, **kwargs):
        self.new_flag, self.convert = new_flag, convert
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        logger.warning("%s is deprecated; use %s", option_string, self.new_flag)
        setattr(namespace, self.dest, self.convert(values) if self.convert else values)


def _parse_bank_readahead(value: str) -> str:
    """--moe-bank-readahead: off, auto, or a positive number of kB."""
    v = str(value).strip().lower()
    if v in ("off", "auto"):
        return v
    try:
        kb = int(v)
    except ValueError:
        kb = 0
    if kb <= 0:
        raise argparse.ArgumentTypeError(f"expected off, auto or a positive kB value, got {value!r}")
    return str(kb)


def _nvfp4_entry(value: str) -> str:
    """The --quant-backend entry an old --nvfp4-backend value stands for; auto stands for none."""
    if value == "auto":
        return ""
    return "moe.nvfp4=" + {"flashinfer": "b12x"}.get(value, value)


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    # The terminal shell is attached to this server (ft shell --model / ft serve --shell-mode).
    # The workers read it to leave the shell's foreground process group, so the ^C that cancels
    # a turn cannot also kill the engine — see server/launch.py:_detach_process_group.
    shell_mode: bool = False
    served_model_name: str | None = None
    tool_call_parser: str = "llama3"
    # Reasoning parser that splits <think> reasoning from content for OpenAI
    # responses. None disables it (default for models without a reasoning protocol).
    reasoning_parser: str | None = None
    # "model": fill unspecified request sampling params from generation_config.json
    # (temperature/top_k/top_p), like sglang. "none": use framework defaults only.
    sampling_defaults: str = "model"
    # Default max output (decode) tokens for a request that omits one. None falls back to the
    # adapter's built-in default (32k).
    max_output_tokens: int | None = None
    # Report the prefix-cache hit in each response's usage block (OpenAI
    # prompt_tokens_details.cached_tokens, Anthropic cache_read_input_tokens, Responses
    # input_tokens_details.cached_tokens). Mirrors sglang's --enable-cache-report.
    enable_cache_report: bool = False
    # Comma-separated hostname allowlist for client-supplied image URLs; empty admits any domain.
    allowed_media_domains: str = ""
    # Directory file:// image refs may be read from; empty rejects local files.
    allowed_local_media_path: str = ""
    # Comma-separated CORS allow-list for browser/webview clients (e.g. the desktop
    # app). Empty string disables CORS headers entirely; "*" allows any origin.
    cors_origins: str = "tauri://localhost,http://tauri.localhost,http://localhost:1420"
    # --gpu entries in TP-rank order, empty = not given
    gpu: tuple[str, ...] = ()
    # full UUIDs resolved from --gpu, entry i = TP rank i; None = NVML unavailable, each worker then resolves its raw entry against CUDA's own enumeration
    gpu_assigned: "tuple[str, ...] | None" = None

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/freetoken_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/freetoken_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def _json_object(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return value


def parse_args(
    args: List[str],
    run_shell: bool = False,
    prog: str | None = None,
) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from freetoken.attention import validate_attn_backend
    from freetoken.kvcache import SUPPORTED_CACHE_MANAGER
    from freetoken.moe import MOE_STRATEGIES

    def _parse_quant_backend(value: str) -> str:
        from freetoken.layers.quantization import QuantBackend

        try:
            QuantBackend.parse(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from None
        return value

    def _parse_moe_cache_rate(value: str) -> float:
        try:
            rate = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
        if not 0 <= rate <= 1:
            raise argparse.ArgumentTypeError("must be in [0, 1]")
        return rate

    def _positive_int(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a positive integer") from exc
        if n < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return n

    def _lazy_gpu_arg(value: str) -> tuple[str, ...]:
        from freetoken.gpu_select import gpu_arg

        return gpu_arg(value)

    def _infer_tool_call_parser(model_path: str) -> str:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        # M3 first: its marker also contains the bare "minimax" substring, but the
        # namespaced tool grammar is a different protocol from M2's.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
            return "qwen3_coder"
        if (
            "qwen3_5" in marker
            or "qwen3.5" in marker
            or ("qwen3" in marker and "coder" in marker)
        ):
            return "qwen3_coder"
        if "qwen" in marker:
            return "qwen25"
        if "deepseek" in marker and ("v4" in marker or "deepseek_v4" in marker):
            return "deepseekv32"
        if "deepseek" in marker and ("v3.2" in marker or "v32" in marker):
            return "deepseekv32"
        if "glm" in marker:
            return "glm47"
        if "mistral" in marker:
            return "mistral"
        return "llama3"

    def _infer_reasoning_parser(model_path: str) -> str | None:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        if "deepseek" in marker and any(
            tag in marker for tag in ("v4", "deepseek_v4", "v3.2", "v32")
        ):
            return "deepseekv32"
        if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
            return "qwen3"
        if "qwen3" in marker or "qwen3.5" in marker or "qwen3_5" in marker:
            return "qwen3"
        if "glm" in marker:
            return "glm"
        # M3 first ("minimax" is a substring): <mm:think> tags + 3 thinking gears,
        # not M2's always-on implicit <think>.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        return None

    parser = argparse.ArgumentParser(prog=prog, description="FreeToken Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--host-embedding",
        action="store_true",
        default=False,
        help=(
            "Keep the input embedding table in pinned host memory; the GPU gathers the rows it "
            "needs in place over PCIe (also inside CUDA graphs). Frees the table's VRAM (about "
            "1 GB for a 250k x 2048 vocabulary) for KV pages on small cards. Qwen3.5-MoE family."
        ),
    )

    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help=(
            "Pipeline (layer-split) parallelism: run the decoder layers as N contiguous "
            "blocks, one process per GPU (--gpu lists them in rank order). Rank 0 owns the "
            "embedding, the last rank owns the head; the residual stream crosses ranks over "
            "gloo, so no NCCL/P2P is needed. Mutually exclusive with --tp-size > 1."
        ),
    )

    parser.add_argument(
        "--dense-quant",
        type=str,
        default="none",
        choices=["none", "fp8"],
        help=(
            "Quantize the checkpoint's bf16 dense (non-expert) weights at load: 'fp8' = per-row "
            "fp8-e4m3 W8A16 for attention, GDN, shared expert, lm_head and the embedding "
            "(roughly halves their VRAM and per-token read traffic). A projection the checkpoint "
            "already quantized keeps its own format, and the router, hyper-connection, QSA indexer, "
            "PLE and GDN b/a gates stay bf16. Measured on qwen4_exp (Qwen3.8-Flash-Next)."
        ),
    )

    parser.add_argument(
        "--spec-mtp",
        type=int,
        default=0,
        help=(
            "MTP speculative decoding: verify K drafts from the checkpoint's own MTP head per "
            "step (0 = off). Single-request decode (use --max-running-req 1); the draft head "
            "runs on the head-owning rank and adds one full-attention layer and one expert-bank "
            "layer (its bf16 experts are quantized to NVFP4 at load). Qwen3.5-MoE family and "
            "Qwen3.8-Flash-Next NVFP4 checkpoints."
        ),
    )

    parser.add_argument(
        "--pp-layers",
        type=str,
        default=None,
        help=(
            "Layer boundaries of the --pp-size split, comma-separated (N-1 values): '24' "
            "gives rank 0 layers [0,24) and rank 1 [24,48). Default: even split."
        ),
    )

    parser.add_argument(
        "--gpu",
        type=_lazy_gpu_arg,
        default=ServerArgs.gpu,
        help=(
            "GPU(s) to run on, comma-separated; entry i is TP rank i. Each entry is a GPU "
            "UUID (GPU-xxxx..., as nvidia-smi -L prints) or an nvidia-smi index"
        ),
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=ServerArgs.max_output_tokens,
        help="Default max output tokens for requests that omit one (default 32k).",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help=(
            "Fraction of total GPU free memory the engine may use for weights + MoE "
            "cache + KV cache combined; the remainder is reserved runtime headroom."
        ),
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--decode-log-interval",
        type=_positive_int,
        default=ServerArgs.decode_log_interval,
        help="Print one decode scheduler status line every N decode forwards.",
    )

    kv_capacity_group = parser.add_mutually_exclusive_group()
    kv_capacity_group.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    kv_capacity_group.add_argument(
        "--num-tokens",
        dest="num_token_override",
        type=int,
        default=ServerArgs.num_token_override,
        help=(
            "Total KV-cache capacity in tokens; must be a multiple of the resolved page "
            "size (DSV4: 128 window page, TRTLLM backend: 64). Mutually exclusive with "
            "--num-pages."
        ),
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="KV cache strategy (naive | radix). For hybrid GDN models 'radix' is materialized "
        "as a GDN-aware radix (cross-request GDN-state prefix reuse); pass 'naive' to opt out.",
    )

    parser.add_argument(
        "--text-model-only",
        action="store_true",
        default=False,
        help="Serve a multimodal checkpoint text-only: no encoder tower is built (its VRAM goes to "
        "the KV/expert pools) and every multimodal input is rejected. Same as --mm-disable with "
        "every encoder kind.",
    )
    parser.add_argument(
        "--mm-disable",
        nargs="+",
        choices=list(ENCODER_KINDS),
        default=[],
        metavar="{vision,audio}",
        help="Encoder towers to leave unbuilt; every input they would serve is rejected.",
    )

    parser.add_argument(
        "--image-min-tokens",
        type=_positive_int,
        default=MultimodalConfig.image_min_tokens,
        help="Fewest tokens an image may take: the image processor scales smaller images up to it, "
        "in the family's own units. Default: the processor's own limit.",
    )
    parser.add_argument(
        "--image-max-tokens",
        type=_positive_int,
        default=MultimodalConfig.image_max_tokens,
        help="Most tokens an image may take: the image processor scales larger images down to it, "
        "in the family's own units (Qwen VL: one token per 32x32 pixels). Default: the processor's own limit.",
    )
    parser.add_argument(
        "--mm-processor-kwargs",
        type=_json_object,
        default=None,
        metavar="JSON",
        help="JSON object of extra keyword arguments for the checkpoint's image processor call, "
        "for family-specific knobs; applied after the token budget.",
    )

    parser.add_argument(
        "--mm-embed-cache-device",
        choices=["cpu", "cuda"],
        default=MultimodalConfig.embed_cache_device,
        help="Storage for encoded image embeddings between prefill chunks.",
    )

    parser.add_argument(
        "--mm-encoder-weights",
        choices=["gpu", "host", "cpu"],
        default=MultimodalConfig.encoder_weights,
        help="Encoder tower block weights: pinned host banks streamed two blocks at a time behind the "
        "compute (default, about 60 MiB of VRAM instead of the whole tower), or resident on the GPU. "
        "cpu: the vision tower runs on the CPU in the tokenizer worker (no VRAM and no pinned memory, "
        "a few seconds per image; the Qwen3.5/3.6 and Qwen3.8-Flash-Next towers).",
    )

    parser.add_argument(
        "--mm-encoder-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default=MultimodalConfig.encoder_dtype,
        help="Compute dtype of the Qwen VL vision tower on the GPU (not --mm-encoder-weights cpu, which is "
        "float32; other families' towers follow the model dtype). auto: "
        "float32 when the model runs bfloat16 -- the Qwen VL vision tower loses about 9%% of its output in "
        "bfloat16 -- otherwise the model dtype.",
    )

    parser.add_argument(
        "--allowed-media-domains",
        type=str,
        default=ServerArgs.allowed_media_domains,
        help="Comma-separated hostname allowlist for client-supplied image URLs. "
        "Empty (default) allows any domain.",
    )

    parser.add_argument(
        "--allowed-local-media-path",
        type=str,
        default=ServerArgs.allowed_local_media_path,
        help="Directory that file:// image refs may be read from. "
        "Unset (default) rejects local files.",
    )

    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        default=ServerArgs.enable_cache_report,
        help=(
            "Return the number of prefix-cached prompt tokens in each response's usage block "
            "(OpenAI usage.prompt_tokens_details.cached_tokens, Anthropic "
            "usage.cache_read_input_tokens, Responses usage.input_tokens_details.cached_tokens). "
            "On /v1/messages this also makes input_tokens EXCLUDE the cached prefix, matching "
            "Anthropic billing semantics."
        ),
    )

    parser.add_argument(
        "--sampling-defaults",
        type=str,
        default=ServerArgs.sampling_defaults,
        choices=["model", "none"],
        help=(
            "Source for unspecified request sampling params. 'model' fills "
            "temperature/top_k/top_p from the checkpoint's generation_config.json "
            "(recommended for reasoning models to avoid greedy repetition loops); "
            "'none' uses framework defaults only."
        ),
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=ServerArgs.served_model_name,
        help="Model id returned by /v1/models. Defaults to the basename of --model.",
    )

    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="auto",
        choices=[
            "auto",
            "llama3",
            "qwen",
            "qwen25",
            "qwen3_coder",
            "mistral",
            "deepseekv32",
            "gemma4",
            "glm47",
            "minimax",
            "minimax_m3",
            "muse_glimmer",
            "gpt_oss",
            "gpt-oss",
        ],
        help="Tool-call parser format for OpenAI-compatible tool responses.",
    )

    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="auto",
        choices=[
            "auto", "off", "deepseekv32", "gpt_oss", "qwen3", "glm",
            "minimax", "minimax_m3", "muse_glimmer", "gemma4",
        ],
        help=(
            "Reasoning parser that splits chain-of-thought into reasoning_content "
            "for OpenAI responses. 'auto' selects per model family (gpt-oss Harmony, "
            "<think> for qwen3/glm/minimax, <mm:think> for minimax-m3, ATEM to=self "
            "channels for muse-glimmer, gemma thought, dsv4); 'off' disables it."
        ),
    )

    parser.add_argument(
        "--moe-strategy",
        default=ServerArgs.moe_strategy,
        choices=["auto", *MOE_STRATEGIES],
        help=(
            "How the routed experts are served. 'auto' resolves a MoE model to the offload family "
            "(offload, or hybrid when a `ft bench bw` profile recommends it); resident "
            "'fused' experts must be requested explicitly."
        ),
    )

    parser.add_argument(
        "--moe-backend",
        dest="moe_strategy",
        action=_DeprecatedAlias,
        new_flag="--moe-strategy",
        default=argparse.SUPPRESS,
        choices=["auto", *MOE_STRATEGIES],
        help="[Deprecated] Use --moe-strategy.",
    )

    parser.add_argument(
        "--quant-backend",
        default=None,
        type=_parse_quant_backend,
        help=(
            "Kernel per quantized layer type: comma-separated layer[.kind]=name entries, e.g. "
            "'linear=marlin,moe=b12x' or 'moe.nvfp4=triton'. A layer-level entry applies to every "
            "kind whose kernel table lists the name; unlisted tables stay automatic."
        ),
    )

    parser.add_argument(
        "--ple-backend",
        default=ServerArgs.ple_backend,
        choices=["pinned", "disk"],
        help=(
            "Where a PLE n-gram table lives. 'disk' (default) reads rows straight from the "
            "checkpoint files; 'pinned' preloads the whole table into page-locked host RAM."
        ),
    )

    parser.add_argument(
        "--nvfp4-backend",
        action=_DeprecatedAlias,
        new_flag="--quant-backend moe.nvfp4=<marlin|b12x|triton>",
        convert=_nvfp4_entry,
        default=argparse.SUPPRESS,
        choices=["auto", "marlin", "flashinfer", "triton"],
        help="[Deprecated] Use --quant-backend moe.nvfp4=<marlin|b12x|triton> ('flashinfer' is b12x).",
    )

    parser.add_argument(
        "--expert-load",
        default=ServerArgs.expert_load,
        choices=["auto", "serial", "parallel"],
        help=(
            "How MoE expert banks are read into host RAM. 'auto' (default) reads scattered "
            "experts in parallel (fast) but falls back to serial when free RAM can't cover "
            "the banks + the parallel reader's extra whole-shard buffer; 'serial' forces the "
            "low-memory reclaimable read (slower); 'parallel' forces the fast read."
        ),
    )

    moe_cache_group = parser.add_mutually_exclusive_group()
    moe_cache_group.add_argument(
        "--moe-cache-size",
        type=int,
        default=ServerArgs.moe_cache_size,
        help="The number of unified MoE expert slots on GPU.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-rate",
        type=_parse_moe_cache_rate,
        default=ServerArgs.moe_cache_rate,
        help="The fraction of all MoE experts to keep in GPU cache.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-auto",
        action="store_true",
        default=ServerArgs.moe_cache_auto,
        help=(
            "Auto-pick --moe-cache-size from free VRAM and expert size, MoE-priority "
            "(KV gets --kv-reserve-tokens as a floor). Not supported for owned-KV models."
        ),
    )

    parser.add_argument(
        "--linear-state-cache-ratio",
        type=float,
        default=ServerArgs.linear_state_cache_ratio,
        help=(
            "Hybrid GDN models (Qwen3.5-MoE, Qwen3.8-Flash-Next): GDN-state snapshots kept for "
            "prefix reuse, per running request, on top of the 4 slots each request needs to "
            "run (floor 4). A prefix can only be resumed from a live snapshot, so this -- not "
            "the KV budget -- bounds how many conversations stay reusable: at the default 2.0 "
            "with --max-running-req 1 the cache holds 4 snapshots, and switching between more "
            "conversations than that re-prefills them. How many snapshots a conversation "
            "costs is per model; see docs/prefix-reuse.md. Each slot is one GDN state in VRAM "
            "(every GDN layer's recurrent + conv state; about 62 MiB for 30 GDN layers)."
        ),
    )
    parser.add_argument(
        "--linear-state-host-slots",
        type=int,
        default=ServerArgs.linear_state_host_slots,
        help=(
            "Hybrid GDN models: pinned host RAM slots for GDN-state snapshots (default 0 = off). "
            "When the VRAM snapshot slots (--linear-state-cache-ratio) run short, the least "
            "recently used snapshot moves here instead of being dropped, and a later hit copies "
            "it back up (a few ms). This raises how many conversations stay reusable without "
            "spending VRAM: each slot costs one GDN state of pinned RAM (about 62 MiB for 30 GDN "
            "layers; per pipeline rank, its own layers only) and counts against the pin budget."
        ),
    )
    parser.add_argument(
        "--prefix-disk-cache",
        default=ServerArgs.prefix_disk_cache,
        metavar="DIR",
        help=(
            "Hybrid GDN models (Qwen3.5-MoE, Qwen3.8-Flash-Next), one GPU: keep prefix-cache "
            "entries -- a prompt's KV pages and the GDN state snapshot at its end -- in this "
            "directory, written while the server is idle, and read them back instead of "
            "prefilling again when a prompt starts with one the in-memory cache no longer "
            "holds, including after a restart. Only prefixes of 1024+ tokens are written. "
            "Entries are keyed by the exact tokens and by the model, weights and cache layout; "
            "anything else is never read. Refused with --pp-size / --tp-size > 1. "
            "See docs/prefix-reuse.md."
        ),
    )
    parser.add_argument(
        "--prefix-disk-cache-size",
        default=ServerArgs.prefix_disk_cache_size,
        metavar="SIZE",
        help=(
            "Cap for --prefix-disk-cache's directory (e.g. 32G, the default). The least "
            "recently used entries are removed to stay under it."
        ),
    )
    parser.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=ServerArgs.kv_reserve_tokens,
        help="KV-cache token floor reserved before --moe-cache-auto fills experts.",
    )

    parser.add_argument(
        "--moe-collect-stats",
        action="store_true",
        default=ServerArgs.moe_collect_stats,
        help=(
            "Accumulate decode miss-rate counters in the offload MoE cache (device-side, "
            "captured into the decode graph). Read back by --moe-stats-out."
        ),
    )
    parser.add_argument(
        "--moe-stats-out",
        default=ServerArgs.moe_stats_out,
        help=(
            "Write the decode routing histogram (per layer, per expert) and the realized "
            "miss rates to this JSON path, rewritten each time the server goes idle and on "
            "shutdown; implies --moe-collect-stats. One "
            "file per pipeline rank (.rank<N>.json) when --pp-size > 1. The histogram is "
            "only accumulated outside a captured graph, so pair it with "
            "--disable-cuda-graph for a collection run."
        ),
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help=(
            "Run decode eagerly (no CUDA graph capture). Much slower; for instrumentation "
            "runs whose counters must see the real per-step routing."
        ),
    )

    parser.add_argument(
        "--pp-send-ahead",
        type=int,
        default=ServerArgs.pp_send_ahead,
        help=(
            "With --pp-size: how many residual streams the first rank may have in flight to the "
            "next rank at once (default 1 = the send waits for the peer's receive, so the rank "
            "stays one chunk ahead). Above 1 it keeps prefilling chunks while the next rank works "
            "through what it has, at one chunk's residual stream of pinned host memory per slot."
        ),
    )

    parser.add_argument(
        "--pp-prefill-group",
        type=int,
        default=ServerArgs.pp_prefill_group,
        help=(
            "Qwen3.8-Flash-Next with --pp-size 2 and offloaded experts: the second rank holds up to this many "
            "consecutive prefill chunks of a prompt and runs them layer by layer, so each layer's "
            "expert bank is copied to its GPU once per group instead of once per chunk "
            "(default 1 = off). For a rank on a slow link (a PCIe x4 slot copies a Flash-Next "
            "rank's banks in about 5.6 s per chunk). Costs one residual stream of VRAM per held "
            "chunk on that rank. Needs --max-running-req 1."
        ),
    )

    parser.add_argument(
        "--prefill-mixer-pieces",
        type=int,
        default=ServerArgs.prefill_mixer_pieces,
        help=(
            "Run each prefill chunk's GDN / attention over this many consecutive pieces and its "
            "MoE over the whole chunk (default 1 = off). The mixers are what cap a chunk's width "
            "on a small card, so with pieces the chunk-budget probe measures less and picks wider "
            "chunks, and an offloaded MoE streams its expert banks fewer times per prompt. "
            "Measured on an RTX 2060 (Ornith, 19.9k-token prompt): 490 -> 649 tok/s at 2, "
            "722 tok/s at 4 with --max-prefill-length 16384. Qwen3.8-Flash-Next (one GPU or "
            "--pp-size) and single-GPU Qwen3.5-MoE."
        ),
    )

    parser.add_argument(
        "--prefill-profile",
        action="store_true",
        default=ServerArgs.prefill_profile,
        help=(
            "Log one line per prefill forward on every rank splitting its wall time into "
            "waiting for the other pipeline rank, host copies of expert rows the GPU cannot "
            "read directly (page faults on a --moe-bank-ram bank land here), per-layer "
            "embedding reads, and the GPU plus the rest; with the GiB copied and their rate, "
            "major page faults, storage reads, how much of the bank the page cache held, and "
            "the achieved PCIe rate of the registered rows. Costs a device sync per forward."
        ),
    )
    parser.add_argument(
        "--prefill-chunk-budget",
        type=float,
        default=ServerArgs.prefill_chunk_budget,
        help=(
            "Share of the free VRAM one prefill chunk's transient may take (default 0.55; "
            "0 disables and --max-prefill-length is used as given). The chunk is what the "
            "linear-attention kernels size their per-forward buffers from, and a chunk whose "
            "transient is the size of the free VRAM runs slow before it runs out; the engine "
            "measures that cost per token at startup and re-solves the chunk before every "
            "prefill against the VRAM free right then. The rest is headroom for whatever else "
            "uses the card: a dedicated box can run 0.8-0.9, a desktop that opens a browser "
            "mid-request wants less."
        ),
    )

    parser.add_argument(
        "--kv-cache-dtype",
        default=ServerArgs.kv_cache_dtype,
        choices=["auto", "q8_0", "q4_0"],
        help=(
            "Store the paged KV cache as block-quantized codes instead of 16-bit: q8_0 is "
            "1.88x smaller, q4_0 3.56x. The point on a small card is not context length but "
            "the expert cache -- --moe-cache-auto hands the freed VRAM to MoE slots, and a "
            "deeper expert cache is what decode time is made of. Plain paged-attention "
            "models, Qwen3.8-Flash-Next and gpt-oss (both its full and its sliding-window "
            "layers); refused at startup otherwise. q4_0 costs measurable accuracy."
        ),
    )

    parser.add_argument(
        "--moe-bank-ram",
        default=ServerArgs.moe_bank_ram,
        help=(
            "Cap host RAM for the expert banks (e.g. 50G), across the whole host: a "
            "--pp-size 2 run splits it between the two ranks. Experts past the cap move to "
            "a cold bank file on disk, chosen by measured routing frequency. Needs an NVMe: "
            "the cold half is read per token. 'auto' takes MemAvailable at startup less "
            "4.5 GiB per rank for the rest of the server and a page cache margin, and logs "
            "the arithmetic. 'ft doctor disk' says whether this host suits it."
        ),
    )
    parser.add_argument(
        "--moe-bank-stats",
        nargs="+",
        default=ServerArgs.moe_bank_stats,
        help=(
            "--moe-stats-out histogram(s) ordering the --moe-bank-ram placement (one file "
            "per pipeline rank). Without them the ordering ignores routing entirely."
        ),
    )
    parser.add_argument(
        "--moe-bank-dir",
        default=ServerArgs.moe_bank_dir,
        help=(
            "Directory for the bank file (bank.ftmb, every MoE layer in one file). Defaults to "
            "the one a packed checkpoint names, else ~/.cache/freetoken/bankmap/<model>."
        ),
    )
    parser.add_argument(
        "--moe-bank-readahead",
        type=_parse_bank_readahead,
        default=ServerArgs.moe_bank_readahead,
        help=(
            "With --moe-bank-ram: the device readahead window, which decides how much a fault "
            "on a non-resident row reads (measured 2.5x on decode). 'off' (default) only logs it "
            "against the model's block geometry; 'auto' writes the recommended window to sysfs "
            "and a number writes that many kB. Needs permission to write "
            "/sys/block/<dev>/queue/read_ahead_kb (otherwise the exact command is logged once); "
            "applies to the whole device and stays set after the server exits. An open mapping "
            "keeps the window it was opened with, so a change by hand needs a restart."
        ),
    )
    parser.add_argument(
        "--moe-bank-rewarm",
        type=float,
        default=ServerArgs.moe_bank_rewarm,
        help=(
            "With --moe-bank-ram: after this many seconds idle, check how much of the bank's "
            "file-backed rows the page cache still holds, and if it has dropped (memory "
            "pressure, WSL2 autoMemoryReclaim), read them back in file order until a request "
            "arrives. Measured on an RTX 2060: the first follow-up after the cache was emptied "
            "took 31 s to its first token instead of 1 s. 0 (default) = off."
        ),
    )
    parser.add_argument(
        "--moe-bank-prefetch",
        action="store_true",
        dest="moe_bank_prefetch",
        default=ServerArgs.moe_bank_prefetch,
        help=(
            "With --moe-bank-ram and cpu/hybrid decode: before the CPU executor computes a "
            "layer, ask the kernel for exactly the non-resident expert rows that layer routes "
            "to (mincore, then MADV_WILLNEED), so the workers wait on reads already in flight "
            "instead of faulting 4 KiB at a time through readahead windows that also read "
            "neighbouring experts. Linux only. Experimental, off by default."
        ),
    )

    parser.add_argument(
        "--moe-cache-policy",
        default=ServerArgs.moe_cache_policy,
        choices=["lru"],
        help="The unified MoE cache eviction policy.",
    )

    parser.add_argument(
        "--moe-cpu-threads",
        type=int,
        default=ServerArgs.moe_cpu_threads,
        help=(
            "Number of CPU worker threads for --moe-strategy cpu decode experts. "
            "0 = auto (physical cores)."
        ),
    )

    parser.add_argument(
        "--moe-cpu-layers",
        type=str,
        default=ServerArgs.moe_cpu_layers,
        help=(
            "With --moe-strategy offload/hybrid: which MoE layers compute on the "
            "CPU executor instead of the GPU offload/PCIe path (where CUDA pinning "
            "is quota-capped, e.g. WSL, their banks are OS-locked instead of pinned). Explicit id list ('3,7,11'), a count ('8' = 8 "
            "layers evenly strided), a fraction ('0.5'), or 'auto'. 'auto' is for Windows/WSL "
            "only, where CUDA pinned memory is capped: it locks just enough head+tail layers "
            "for the banks over the pin budget. Any value, 'auto' included, commits to CPU "
            "decode before the model is built, so the expert format must have a CPU executor "
            "path (bf16, nvfp4, mxfp4); do not pass it on Linux. Unset = every layer on the "
            "GPU; a boot whose banks exceed a known pin budget stops and asks for this flag."
        ),
    )

    parser.add_argument(
        "--moe-hybrid-max-fetch",
        type=int,
        default=ServerArgs.moe_hybrid_max_fetch,
        help=(
            "For --moe-strategy hybrid: max experts fetched over PCIe per (layer, decode "
            "step); the rest of that step's misses are computed on the CPU, overlapped. "
            "-1 (default) = auto: fetch the benched pcie/cpu bandwidth fraction of each "
            "step's misses (perfect overlap; needs an `ft bench bw` profile, else 1). "
            "0 = never fetch (all misses on CPU); large = behaves like plain offload."
        ),
    )

    parser.add_argument(
        "--disable-moe-prefill-overlap",
        action="store_false",
        dest="moe_prefill_overlap",
        default=ServerArgs.moe_prefill_overlap,
        help=(
            "Disable two-buffer overlap for prefill MoE expert copies. "
            "By default, prefill overlap is enabled and requires "
            "--moe-cache-size >= 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--enable-special-token-ckpt",
        action="store_true",
        dest="special_token_ckpt",
        default=ServerArgs.special_token_ckpt,
        help=(
            "Checkpoint decode state at special tokens (currently the tool-call opener). "
            "When a GDN-hybrid or SWA model samples its tool-call opener token, the "
            "scheduler preserves a reuse point just after it (GDN: a state snapshot "
            "donated to the prefix cache; SWA: the trailing window is kept resumable), so "
            "a client that rewrites the echoed tool call only invalidates the call body, "
            "not the turn."
        ),
    )

    parser.add_argument(
        "--moe-prefill-hit-d2d",
        action="store_true",
        dest="moe_prefill_hit_d2d",
        default=ServerArgs.moe_prefill_hit_d2d,
        help=(
            "During prefill prefetch, copy cache-resident experts device-side into "
            "the double buffer and stream only the misses over PCIe "
            "(cudaMemcpyBatchAsync, CUDA >= 13.0). Effective with "
            "--moe-cache-size > 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--cors-origins",
        type=str,
        default=ServerArgs.cors_origins,
        help=(
            "Comma-separated CORS allow-list for browser/webview clients "
            "(default: local Tauri/Vite dev origins). '' disables, '*' allows any."
        ),
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # --pp-size N: the N ranks split the layers; tp_info carries the world (rank, size) as
    # for TP, `parallel` tells the engine how to use it.
    pp_size = kwargs.pop("pp_size")
    pp_layers = kwargs.pop("pp_layers")
    kwargs["pp_split"] = None
    if pp_size > 1:
        if kwargs["tensor_parallel_size"] > 1:
            parser.error("--pp-size and --tp-size cannot both be > 1")
        kwargs["parallel"] = "pp"
        kwargs["tensor_parallel_size"] = pp_size
        if pp_layers:
            try:
                split = tuple(int(x) for x in pp_layers.split(",") if x.strip())
            except ValueError:
                parser.error(f"--pp-layers must be comma-separated integers, got {pp_layers!r}")
            if len(split) != pp_size - 1 or any(b <= a for a, b in zip(split, split[1:])):
                parser.error(
                    f"--pp-layers needs {pp_size - 1} strictly increasing boundaries for "
                    f"--pp-size {pp_size}, got {pp_layers!r}"
                )
            kwargs["pp_split"] = split
    elif pp_layers:
        parser.error("--pp-layers needs --pp-size > 1")
    if kwargs["pp_send_ahead"] < 1:
        parser.error("--pp-send-ahead must be >= 1")
    if kwargs["pp_send_ahead"] > 1 and pp_size < 2:
        parser.error("--pp-send-ahead needs --pp-size > 1")
    if kwargs["pp_prefill_group"] < 1:
        parser.error("--pp-prefill-group must be >= 1")
    if kwargs["pp_prefill_group"] > 1:
        if pp_size != 2:
            parser.error("--pp-prefill-group needs --pp-size 2")
        if kwargs["max_running_req"] != 1:
            parser.error("--pp-prefill-group needs --max-running-req 1")

    if kwargs["prefix_disk_cache"]:
        if kwargs["tensor_parallel_size"] > 1:
            parser.error(
                "--prefix-disk-cache runs on one GPU only for now: it is refused with "
                f"{'--pp-size' if pp_size > 1 else '--tp-size'} > 1 (every rank would have to "
                "restore the same prefix at the same step)"
            )
        from freetoken.moe.bank_disk import parse_size

        try:
            size = parse_size(kwargs["prefix_disk_cache_size"])
        except ValueError as exc:
            parser.error(f"--prefix-disk-cache-size: {exc}")
        if not size or size <= 0:
            parser.error("--prefix-disk-cache-size must be a positive size, e.g. 32G")

    # reject a too-long list here with a clear reason, not as a dead rank later
    if len(kwargs["gpu"]) not in (0, kwargs["tensor_parallel_size"]):
        if kwargs["tensor_parallel_size"] == 1 and len(kwargs["gpu"]) > 1:
            parser.error(
                "tensor parallelism is not supported yet: --gpu takes one entry "
                "(give --pp-size N to split the layers over N GPUs)"
            )
        parser.error(
            f"--gpu has {len(kwargs['gpu'])} entries but --tensor-parallel-size is "
            f"{kwargs['tensor_parallel_size']}; give one entry per TP rank"
        )

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    kwargs["shell_mode"] = run_shell
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    # the old flag stands in for one --quant-backend entry; next to the real flag it is a usage error
    entry = kwargs.pop("nvfp4_backend", None)
    if entry is not None:
        if kwargs["quant_backend"] is not None:
            parser.error("--nvfp4-backend cannot be combined with --quant-backend; write --quant-backend moe.nvfp4=... instead")
        if entry:
            kwargs["quant_backend"] = entry

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    # a bad media root is a deployment mistake; fail at startup, not per request
    if kwargs["allowed_local_media_path"]:
        media_root = os.path.realpath(os.path.expanduser(kwargs["allowed_local_media_path"]))
        if not os.path.isdir(media_root):
            parser.error(f"--allowed-local-media-path {media_root} is not a directory")
        kwargs["allowed_local_media_path"] = media_root

    if kwargs["served_model_name"] is None:
        kwargs["served_model_name"] = (
            os.path.basename(os.path.normpath(kwargs["model_path"])) or kwargs["model_path"]
        )

    if kwargs["tool_call_parser"] == "auto":
        kwargs["tool_call_parser"] = _infer_tool_call_parser(kwargs["model_path"])

    if kwargs["reasoning_parser"] == "auto":
        kwargs["reasoning_parser"] = _infer_reasoning_parser(kwargs["model_path"])
    elif kwargs["reasoning_parser"] == "off":
        kwargs["reasoning_parser"] = None

    # --disable-cuda-graph is spelled as its own flag rather than exposing cuda_graph_bs:
    # an empty bs list is already the "graphs off" contract downstream (CudaGraphRunner
    # returns early on max_graph_bs == 0), and a list-valued CLI flag would invite
    # half-disabled states.
    if kwargs.pop("disable_cuda_graph", False):
        kwargs["cuda_graph_bs"] = []

    # --moe-stats-out is the only reader of the miss-rate counters, so asking for the dump
    # is asking for the counters; requiring both flags would only produce empty files.
    if kwargs.get("moe_stats_out"):
        kwargs["moe_collect_stats"] = True

    # Fail on a malformed size here rather than deep in the loader, after the weights have
    # been read.
    if kwargs.get("moe_bank_ram"):
        from freetoken.moe.bank_disk import parse_size

        if str(kwargs["moe_bank_ram"]).strip().lower() == "auto":
            # Resolved here, once, in the launcher: the ranks start together, and each reading
            # MemAvailable while the others allocate would split different numbers.
            from freetoken.moe import disk_probe

            try:
                auto = disk_probe.auto_bank_ram(disk_probe.meminfo(), kwargs["tensor_parallel_size"])
            except ValueError as exc:
                parser.error(str(exc))
            logger.info(auto.reason())
            kwargs["moe_bank_ram"] = auto.as_flag()
        parse_size(kwargs["moe_bank_ram"])

    # Offload-family backends (offload/cpu/hybrid) need a slot cache; if the user gave no
    # sizing flag at all, default to --moe-cache-auto so a bare `ft serve <FTW MoE>` works
    # out of the box (the scheduler resolves the size from free VRAM). Explicit
    # size/rate/auto is preserved.
    from freetoken.moe import is_offload_moe_strategy

    _no_cache_flag = (
        kwargs["moe_cache_size"] == 0
        and not kwargs["moe_cache_auto"]
        and (kwargs["moe_cache_rate"] is None or kwargs["moe_cache_rate"] == 0)
    )
    if is_offload_moe_strategy(kwargs["moe_strategy"]) and _no_cache_flag:
        kwargs["moe_cache_auto"] = True

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # "auto" (or an unspecified dtype) resolves to the checkpoint's dtype. Multimodal /
    # hybrid configs (e.g. Qwen3.5-MoE) keep it under ``text_config`` and use the newer
    # ``dtype`` key rather than top-level ``torch_dtype``, so check both; default bf16.
    if (dtype_str := kwargs["dtype"]) in ("auto", None):
        from freetoken.utils import cached_load_hf_config

        cfg = cached_load_hf_config(kwargs["model_path"]).to_dict()
        text_cfg = cfg.get("text_config") or {}
        dtype_str = (
            cfg.get("torch_dtype") or cfg.get("dtype")
            or text_cfg.get("torch_dtype") or text_cfg.get("dtype") or "bfloat16"
        )

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    disabled = set(ENCODER_KINDS) if kwargs.pop("text_model_only") else set()
    disabled.update(kwargs.pop("mm_disable"))
    image_min_tokens, image_max_tokens = kwargs.pop("image_min_tokens"), kwargs.pop("image_max_tokens")
    if image_min_tokens is not None and image_max_tokens is not None and image_min_tokens > image_max_tokens:
        parser.error(f"--image-min-tokens {image_min_tokens} exceeds --image-max-tokens {image_max_tokens}")
    kwargs["mm"] = MultimodalConfig(
        disabled_encoders=frozenset(disabled),
        embed_cache_device=kwargs.pop("mm_embed_cache_device"),
        encoder_weights=kwargs.pop("mm_encoder_weights"),
        encoder_dtype=kwargs.pop("mm_encoder_dtype"),
        image_min_tokens=image_min_tokens,
        image_max_tokens=image_max_tokens,
        processor_kwargs=kwargs.pop("mm_processor_kwargs") or {},
    )
    result = ServerArgs(**kwargs)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
