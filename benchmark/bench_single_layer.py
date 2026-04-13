#!/usr/bin/env python3
"""
Benchmark a single decoder layer or a local decoder layer window with SGLang's
offline ModelRunner path.

This script benchmarks structure-level inference cost without launching the
full tokenizer/scheduler server stack. It is intended for measuring the
end-to-end cost of attention plus MLP/MoE for one selected decoder layer or a
rebased local layer window.

The script supports TP / DP / EP and several MoE all2all backend choices. Layer
window selection is implemented by loading the original Hugging Face config via
SGLang's `get_config()`, projecting the text config into a local window, and
writing the projected result to a temporary local model directory with a
projected `config.json`.

Example models that this script is designed to handle include:
  - `zai-org/GLM-5`
  - `MiniMaxAI/MiniMax-M2.5`
  - `deepseek-ai/DeepSeek-V3.2-Exp`

Example commands:
Launcher-side DP benchmark with `torchrun --nproc-per-node=dp`:
  torchrun --nproc-per-node=2 benchmark/bench_single_decoder_layer.py \
    --model-path deepseek-ai/DeepSeek-V3.2-Exp \
    --distributed-executor-backend external_launcher \
    --dp 2 \
    --tp 4 \
    --batch-size 64 \
    --seq-len 1024 \
    --output-len 1 \
    --num-layers 2 \
    --layer-start 12 \
    --load-format dummy

Native DP-attention benchmark inside one TP world:
  python3 benchmark/bench_single_decoder_layer.py \
    --model-path deepseek-ai/DeepSeek-V3.2-Exp \
    --tp 8 \
    --dp 8 \
    --enable-dp-attention \
    --all2all-backend deepep_high_throughput \
    --moe-runner-backend deep_gemm \
    --fp8-gemm-backend deep_gemm \
    --batch-size 64 \
    --seq-len 1024 \
    --output-len 1 \
    --num-layers 1 \
    --layer-start 0

# DeepSeek V32 nsys profile
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SGLANG_DEEPEP_SYNC_FINISH=1 \
nsys profile \
    --trace-fork-before-exec=true \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -f true \
    -o sglang_dpsk_tp8_b64_s1024_deepgemm_deepep_nvtx \
    python3 benchmark/bench_single_layer.py \
      --model-path deepseek-ai/DeepSeek-V3.2-Exp \
      --trust-remote-code \
      --tp 8 \
      --all2all-backend deepep_high_throughput \
      --moe-runner-backend deep_gemm \
      --fp8-gemm-backend deep_gemm \
      --batch-size 64 \
      --seq-len 1024 \
      --output-len 1 \
      --num-layers 1 \
      --layer-start 4 \
      --load-format dummy \
      --profile \
      --enable-layerwise-nvtx-marker

# MiniMax-M2.5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SGLANG_DEEPEP_SYNC_FINISH=1 \
nsys profile \
    --trace-fork-before-exec=true \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -f true \
    -o sglang_minimax_tp8_b64_s1024_deepgemm_deepep_uniform \
    python3 benchmark/bench_single_layer.py \
        --model-path MiniMaxAI/MiniMax-M2.5 \
        --trust-remote-code \
        --tp 8 \
        --all2all-backend deepep_high_throughput \
        --moe-runner-backend deep_gemm \
        --fp8-gemm-backend deep_gemm \
        --dtype bfloat16 \
        --batch-size 64 \
        --seq-len 1024 \
        --output-len 1 \
        --num-layers 1 \
        --layer-start 4 \
        --load-format dummy \
        --profile \
        --enable-layerwise-nvtx-marker \
        --moe-router-mode uniform_rank

# glm5-fp8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SGLANG_DEEPEP_SYNC_FINISH=1 \
nsys profile \
    --trace-fork-before-exec=true \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -f true \
    -o sglang_glm5_tp8_b64_s1024_deepgemm_deepep_nvtx \
    python3 benchmark/bench_single_layer.py \
      --model-path zai-org/GLM-5-FP8 \
      --trust-remote-code \
      --tp 8 \
      --all2all-backend deepep_high_throughput \
      --moe-runner-backend deep_gemm \
      --fp8-gemm-backend deep_gemm \
      --batch-size 64 \
      --seq-len 1024 \
      --output-len 1 \
      --num-layers 1 \
      --layer-start 4 \
      --load-format dummy \
      --profile \
      --enable-layerwise-nvtx-marker


Important launcher note:
  In this SGLang benchmark, `external_launcher` expects `WORLD_SIZE == dp`.
  Each `torchrun` rank acts as one DP launcher and internally starts its own TP
  workers through the local ModelRunner benchmark path. This is intentionally different from
  the vLLM-style `WORLD_SIZE == tp * dp` assumption.

Important DP note:
  `--enable-dp-attention` uses SGLang's native DP-attention layout inside a
  single TP world (`tp` is still the total world size seen by ModelRunner).
  This is different from `external_launcher`, which measures multiple
  launcher-side replicas with per-rank batch sharding.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import multiprocessing as mp
import os
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from datetime import timedelta
from types import MethodType
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.distributed as dist

from sglang.bench_one_batch import (
    decode as bench_decode,
    extend as bench_extend,
    start_profile,
    stop_profile,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    FP8_GEMM_RUNNER_BACKEND_CHOICES,
    MOE_RUNNER_BACKEND_CHOICES,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import (
    configure_logger,
    maybe_reindex_device_id,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_config
from sglang.srt.utils.network import find_process_using_port, is_port_available

PORT_STRIDE = 1000
NCCL_PORT_OFFSET = 1
TORCHRUN_ENV_VARS_TO_CLEAR = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_USE_AGENT_STORE",
)


@dataclass(frozen=True)
class A2ABackendSpec:
    user_choice: str
    moe_a2a_backend: str
    deepep_mode: str


@dataclass
class PreparedModelConfig:
    architecture: str
    model_type: str
    text_config_path: list[str]
    original_num_layers: Optional[int]
    effective_num_layers: int
    vocab_size: int
    bos_token_id: Optional[int]
    special_token_ids: list[int]
    is_moe: bool
    temp_model_dir: str
    temp_config_path: str
    override_notes: str
    layer_start: int
    selected_layer_types: Optional[list[Any]] = None
    original_first_k_dense_replace: Optional[int] = None
    projected_first_k_dense_replace: Optional[int] = None
    original_moe_layer_freq: Optional[Any] = None
    projected_moe_layer_freq: Optional[Any] = None
    original_moe_layers: Optional[list[int]] = None
    projected_moe_layers: Optional[list[int]] = None
    original_moe_layers_enum: Optional[list[int]] = None
    projected_moe_layers_enum: Optional[list[int]] = None
    original_interleave_moe_layer_step: Optional[int] = None
    projected_interleave_moe_layer_step: Optional[int] = None
    single_layer_kind: Optional[str] = None
    selection_notes: list[str] = field(default_factory=list)

    @property
    def text_config_path_str(self) -> str:
        return ".".join(self.text_config_path) if self.text_config_path else "<root>"


@dataclass
class RuntimeParallelConfig:
    tp_size: int
    ep_size: int
    moe_a2a_backend: str
    deepep_mode: str
    notes: list[str] = field(default_factory=list)


def get_default_page_size(model_path: str) -> Optional[int]:
    lower = model_path.lower()
    if "qwen/qwen3.5" in lower or "qwen3.5" in lower:
        return 32
    return None


def _get_child(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def resolve_text_config_object_path(config: Any) -> tuple[list[str], Any]:
    candidate_paths = [
        ["thinker_config", "text_config"],
        ["thinker_config"],
        ["llm_config"],
        ["language_config"],
        ["text_config"],
        [],
    ]
    for path in candidate_paths:
        current = config
        found = True
        for key in path:
            current = _get_child(current, key)
            if current is None:
                found = False
                break
        if found:
            return path, current
    raise ValueError("Failed to resolve text config object path from HF config.")


def get_nested_mapping(root: dict, path: list[str]) -> dict:
    current: Any = root
    for key in path:
        if not isinstance(current, dict):
            joined = ".".join(path)
            raise TypeError(
                f"Expected dict while traversing '{joined}', but got {type(current).__name__}."
            )
        if key not in current:
            joined = ".".join(path)
            raise KeyError(f"Missing key '{key}' while traversing '{joined}'.")
        current = current[key]
    if not isinstance(current, dict):
        joined = ".".join(path) if path else "<root>"
        raise TypeError(
            f"Resolved mapping at '{joined}' is not a dict: {type(current).__name__}."
        )
    return current


def infer_num_hidden_layers(config_obj: Any, config_dict: dict) -> Optional[int]:
    candidates = [
        getattr(config_obj, "num_hidden_layers", None),
        getattr(config_obj, "num_layers", None),
        config_dict.get("num_hidden_layers"),
        config_dict.get("num_layers"),
    ]
    for value in candidates:
        if isinstance(value, int):
            return value

    for key in (
        "layer_types",
        "layers_block_type",
        "hybrid_layer_pattern",
        "moe_layer_freq",
    ):
        value = config_dict.get(key)
        if isinstance(value, (list, tuple)):
            return len(value)

    return None


def is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def looks_like_per_layer_field(key: str) -> bool:
    return (
        "layer" in key
        or key.endswith("_pattern")
        or key.endswith("_types")
        or key.endswith("_type")
        or key.endswith("_blocks")
    )


def slice_like(value: Sequence[Any], start: int, end: int) -> Any:
    sliced = list(value[start:end])
    if isinstance(value, tuple):
        return tuple(sliced)
    return sliced


def _normalize_layer_type_name(value: Any) -> str:
    return str(value).strip().lower()


def has_kv_cache_layers(selected_layer_types: Optional[Sequence[Any]]) -> bool:
    if not selected_layer_types:
        return True
    return any(_normalize_layer_type_name(x) in {"attention", "full_attention"} for x in selected_layer_types)


def has_linear_state_layers(selected_layer_types: Optional[Sequence[Any]]) -> bool:
    if not selected_layer_types:
        return False
    return any(_normalize_layer_type_name(x) == "linear_attention" for x in selected_layer_types)


def project_full_attention_interval(
    selected_layer_types: Sequence[Any],
) -> Optional[int]:
    normalized = [_normalize_layer_type_name(x) for x in selected_layer_types]
    if not normalized:
        return None
    if any(x not in {"attention", "linear_attention"} for x in normalized):
        return None

    attention_positions = [idx for idx, value in enumerate(normalized) if value == "attention"]
    if not attention_positions:
        # Any interval larger than the local window keeps all local layers as linear_attention.
        return len(normalized) + 1

    interval = attention_positions[0] + 1
    expected = [
        "attention" if (idx + 1) % interval == 0 else "linear_attention"
        for idx in range(len(normalized))
    ]
    return interval if expected == normalized else None


def parse_layer_id_list(value: Any, field_name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [int(part.strip()) for part in stripped.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, int):
        return [value]
    raise TypeError(
        f"Unsupported type for explicit layer-id field '{field_name}': {type(value).__name__}."
    )


def restore_layer_id_container(original: Any, values: list[int]) -> Any:
    if isinstance(original, str):
        return ",".join(str(x) for x in values)
    if isinstance(original, tuple):
        return tuple(values)
    if isinstance(original, list):
        return list(values)
    if isinstance(original, int):
        if len(values) != 1:
            raise ValueError(
                f"Cannot restore multiple values {values} into scalar field {original}."
            )
        return values[0]
    return list(values)


def project_first_k_dense_replace(
    first_k_dense_replace: int,
    moe_layer_freq: int,
    layer_start: int,
) -> int:
    if moe_layer_freq <= 0:
        raise ValueError(f"moe_layer_freq must be positive, got {moe_layer_freq}.")
    if layer_start <= first_k_dense_replace:
        return first_k_dense_replace - layer_start
    delta = layer_start - first_k_dense_replace
    return (moe_layer_freq - (delta % moe_layer_freq)) % moe_layer_freq


def is_moe_layer(
    layer_idx: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
) -> bool:
    if moe_layer_freq <= 0:
        return False
    if layer_idx < first_k_dense_replace:
        return False
    return (layer_idx - first_k_dense_replace) % moe_layer_freq == 0


def project_explicit_moe_layers(
    moe_layers: Any,
    layer_start: int,
    num_layers: int,
) -> list[int]:
    end = layer_start + num_layers
    projected = []
    for layer_id in parse_layer_id_list(moe_layers, "moe_layers"):
        if layer_start <= layer_id < end:
            projected.append(layer_id - layer_start)
    return projected


def project_periodic_one_based_pattern(
    interleave_moe_layer_step: int,
    layer_start: int,
    num_layers: int,
) -> int:
    if interleave_moe_layer_step < 0:
        raise ValueError(
            f"interleave_moe_layer_step must be non-negative, got {interleave_moe_layer_step}."
        )
    if interleave_moe_layer_step == 0:
        return 0

    original_pattern = [
        ((layer_start + idx + 1) % interleave_moe_layer_step) == 0
        for idx in range(num_layers)
    ]

    for candidate in range(1, num_layers + 2):
        candidate_pattern = [((idx + 1) % candidate) == 0 for idx in range(num_layers)]
        if candidate_pattern == original_pattern:
            return candidate

    raise ValueError(
        "Cannot exactly project interleave_moe_layer_step="
        f"{interleave_moe_layer_step} for layer window [{layer_start}, {layer_start + num_layers})."
    )


def project_layer_window(
    text_config_dict: dict,
    layer_start: int,
    num_layers: int,
    original_num_layers: Optional[int],
) -> dict[str, Any]:
    if layer_start < 0:
        raise ValueError(f"layer_start must be >= 0, got {layer_start}.")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}.")
    if original_num_layers is not None and layer_start + num_layers > original_num_layers:
        raise ValueError(
            f"Requested layer window [{layer_start}, {layer_start + num_layers}) exceeds total layer count "
            f"{original_num_layers}."
        )

    if original_num_layers is None:
        raise ValueError(
            "Unable to infer original hidden layer count from the resolved text config."
        )

    end = layer_start + num_layers
    updated_keys: set[str] = set()
    notes: list[str] = []
    selected_layer_types: Optional[list[Any]] = None

    explicit_id_fields = [
        "moe_layers",
        "mlp_only_layers",
        "full_attention_layer_ids",
        "swa_attention_layer_ids",
    ]

    original_first_k = text_config_dict.get("first_k_dense_replace")
    original_moe_layer_freq = text_config_dict.get("moe_layer_freq")
    projected_first_k = None
    projected_moe_layer_freq: Any = None
    if isinstance(original_first_k, int) and isinstance(original_moe_layer_freq, int):
        projected_first_k = project_first_k_dense_replace(
            original_first_k, original_moe_layer_freq, layer_start
        )
        projected_moe_layer_freq = original_moe_layer_freq
        original_mask = [
            is_moe_layer(layer_start + idx, original_first_k, original_moe_layer_freq)
            for idx in range(num_layers)
        ]
        projected_mask = [
            is_moe_layer(idx, projected_first_k, original_moe_layer_freq)
            for idx in range(num_layers)
        ]
        if original_mask != projected_mask:
            raise ValueError(
                "Cannot exactly project first_k_dense_replace/moe_layer_freq for "
                f"layer window [{layer_start}, {end}). "
                f"Original mask={original_mask}, projected mask={projected_mask}."
            )
        text_config_dict["first_k_dense_replace"] = projected_first_k
        updated_keys.add("first_k_dense_replace")
        notes.append(
            f"first_k_dense_replace {original_first_k} -> {projected_first_k} with moe_layer_freq={original_moe_layer_freq}"
        )

    original_moe_layers = None
    projected_moe_layers = None
    if "moe_layers" in text_config_dict and text_config_dict.get("moe_layers") is not None:
        original_moe_layers = parse_layer_id_list(text_config_dict["moe_layers"], "moe_layers")
        projected_moe_layers = project_explicit_moe_layers(
            text_config_dict["moe_layers"], layer_start, num_layers
        )
        text_config_dict["moe_layers"] = restore_layer_id_container(
            text_config_dict["moe_layers"], projected_moe_layers
        )
        updated_keys.add("moe_layers")
        notes.append(f"moe_layers {original_moe_layers} -> {projected_moe_layers}")

    original_moe_layers_enum = None
    projected_moe_layers_enum = None
    if (
        "moe_layers_enum" in text_config_dict
        and text_config_dict.get("moe_layers_enum") is not None
    ):
        original_moe_layers_enum = parse_layer_id_list(
            text_config_dict["moe_layers_enum"], "moe_layers_enum"
        )
        projected_moe_layers_enum = project_explicit_moe_layers(
            text_config_dict["moe_layers_enum"], layer_start, num_layers
        )
        text_config_dict["moe_layers_enum"] = restore_layer_id_container(
            text_config_dict["moe_layers_enum"], projected_moe_layers_enum
        )
        updated_keys.add("moe_layers_enum")
        notes.append(
            f"moe_layers_enum {original_moe_layers_enum} -> {projected_moe_layers_enum}"
        )

    original_interleave = text_config_dict.get("interleave_moe_layer_step")
    projected_interleave = None
    if isinstance(original_interleave, int):
        projected_interleave = project_periodic_one_based_pattern(
            original_interleave, layer_start, num_layers
        )
        text_config_dict["interleave_moe_layer_step"] = projected_interleave
        updated_keys.add("interleave_moe_layer_step")
        notes.append(
            f"interleave_moe_layer_step {original_interleave} -> {projected_interleave}"
        )

    for field_name in explicit_id_fields:
        if field_name in updated_keys:
            continue
        value = text_config_dict.get(field_name)
        if value is None:
            continue
        projected = project_explicit_moe_layers(value, layer_start, num_layers)
        text_config_dict[field_name] = restore_layer_id_container(value, projected)
        updated_keys.add(field_name)
        if field_name == "mlp_only_layers":
            notes.append(f"mlp_only_layers -> {projected}")
        elif field_name in {"full_attention_layer_ids", "swa_attention_layer_ids"}:
            notes.append(f"{field_name} -> {projected}")

    for key, value in list(text_config_dict.items()):
        if key in updated_keys:
            continue
        if not looks_like_per_layer_field(key):
            continue
        if not is_sequence(value) or len(value) != original_num_layers:
            continue
        text_config_dict[key] = slice_like(value, layer_start, end)
        updated_keys.add(key)
        if key in {"layer_types", "layers_block_type"}:
            selected_layer_types = list(text_config_dict[key])

    if selected_layer_types is None:
        layer_types = text_config_dict.get("layer_types")
        if is_sequence(layer_types):
            selected_layer_types = list(layer_types)
        else:
            layers_block_type = text_config_dict.get("layers_block_type")
            if is_sequence(layers_block_type):
                selected_layer_types = list(layers_block_type)

    text_config_dict["num_hidden_layers"] = num_layers
    updated_keys.add("num_hidden_layers")
    if "num_layers" in text_config_dict and isinstance(text_config_dict["num_layers"], int):
        text_config_dict["num_layers"] = num_layers
        updated_keys.add("num_layers")

    return {
        "updated_keys": sorted(updated_keys),
        "selection_notes": notes,
        "selected_layer_types": selected_layer_types,
        "original_first_k_dense_replace": original_first_k
        if isinstance(original_first_k, int)
        else None,
        "projected_first_k_dense_replace": projected_first_k,
        "original_moe_layer_freq": original_moe_layer_freq,
        "projected_moe_layer_freq": projected_moe_layer_freq,
        "original_moe_layers": original_moe_layers,
        "projected_moe_layers": projected_moe_layers,
        "original_moe_layers_enum": original_moe_layers_enum,
        "projected_moe_layers_enum": projected_moe_layers_enum,
        "original_interleave_moe_layer_step": original_interleave
        if isinstance(original_interleave, int)
        else None,
        "projected_interleave_moe_layer_step": projected_interleave,
    }


def mirror_projected_fields_to_ancestors(
    root_dict: dict,
    text_config_path: list[str],
    text_config_dict: dict,
    updated_keys: Iterable[str],
) -> None:
    for prefix_len in range(len(text_config_path)):
        ancestor = get_nested_mapping(root_dict, text_config_path[:prefix_len])
        for key in updated_keys:
            if key in ancestor:
                ancestor[key] = copy.deepcopy(text_config_dict.get(key))


def collect_special_token_ids(root_config: Any, text_config: Any) -> list[int]:
    special_ids: set[int] = set()
    for config_obj in (root_config, text_config):
        if config_obj is None:
            continue
        mapping = (
            config_obj.to_dict()
            if hasattr(config_obj, "to_dict")
            else dict(config_obj)
            if isinstance(config_obj, dict)
            else {}
        )
        for key, value in mapping.items():
            if not key.endswith("token_id") and not key.endswith("token_ids"):
                continue
            if isinstance(value, int):
                special_ids.add(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, int):
                        special_ids.add(item)
    return sorted(x for x in special_ids if x >= 0)


def detect_is_moe(config_obj: Any, config_dict: dict) -> bool:
    for key in (
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
        "moe_num_experts",
    ):
        value = getattr(config_obj, key, config_dict.get(key))
        if isinstance(value, int) and value > 0:
            return True

    if isinstance(config_dict.get("interleave_moe_layer_step"), int):
        return True

    if config_dict.get("moe_layers") is not None:
        return len(parse_layer_id_list(config_dict["moe_layers"], "moe_layers")) > 0

    if config_dict.get("moe_layers_enum") is not None:
        return (
            len(parse_layer_id_list(config_dict["moe_layers_enum"], "moe_layers_enum"))
            > 0
        )

    first_k = config_dict.get("first_k_dense_replace")
    moe_layer_freq = config_dict.get("moe_layer_freq")
    if isinstance(first_k, int) and (
        isinstance(moe_layer_freq, int) or isinstance(moe_layer_freq, (list, tuple))
    ):
        return True

    return False


def infer_vocab_size(root_config: Any, text_config: Any, text_config_dict: dict) -> int:
    for source in (
        getattr(text_config, "vocab_size", None),
        getattr(root_config, "vocab_size", None),
        text_config_dict.get("vocab_size"),
    ):
        if isinstance(source, int) and source > 0:
            return source
    raise ValueError("Failed to resolve vocab_size from the projected config.")

def infer_bos_token_id(root_config: Any, text_config: Any) -> Optional[int]:
    for source in (
        getattr(text_config, "bos_token_id", None),
        getattr(root_config, "bos_token_id", None),
    ):
        if isinstance(source, int) and source >= 0:
            return source
    return None


def infer_single_layer_kind(model_cfg: PreparedModelConfig) -> Optional[str]:
    if model_cfg.effective_num_layers != 1:
        return None
    if model_cfg.projected_moe_layers is not None:
        return "moe" if 0 in model_cfg.projected_moe_layers else "dense"
    if model_cfg.projected_moe_layers_enum is not None:
        return "moe" if 0 in model_cfg.projected_moe_layers_enum else "dense"
    if model_cfg.projected_interleave_moe_layer_step is not None:
        if model_cfg.projected_interleave_moe_layer_step == 0:
            return "moe"
        return (
            "moe"
            if (1 % model_cfg.projected_interleave_moe_layer_step) == 0
            else "dense"
        )
    if (
        model_cfg.projected_first_k_dense_replace is not None
        and isinstance(model_cfg.projected_moe_layer_freq, int)
    ):
        return (
            "moe"
            if is_moe_layer(
                0,
                model_cfg.projected_first_k_dense_replace,
                model_cfg.projected_moe_layer_freq,
            )
            else "dense"
        )
    if model_cfg.is_moe:
        return "moe"
    return "dense"


def register_temp_path_for_cleanup(path: str) -> None:
    def _cleanup() -> None:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except FileNotFoundError:
            pass

    atexit.register(_cleanup)


def prepare_model_config(args: argparse.Namespace) -> PreparedModelConfig:
    hf_config = get_config(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    text_config_path, text_config_obj = resolve_text_config_object_path(hf_config)
    config_dict = copy.deepcopy(hf_config.to_dict())
    text_config_dict = get_nested_mapping(config_dict, text_config_path)
    original_num_layers = infer_num_hidden_layers(text_config_obj, text_config_dict)

    projection_info = project_layer_window(
        text_config_dict=text_config_dict,
        layer_start=args.layer_start,
        num_layers=args.num_layers,
        original_num_layers=original_num_layers,
    )
    selected_layer_types = projection_info["selected_layer_types"]
    if selected_layer_types is None and original_num_layers is not None:
        for attr_name in ("layer_types", "layers_block_type"):
            attr_value = getattr(text_config_obj, attr_name, None)
            if is_sequence(attr_value) and len(attr_value) == original_num_layers:
                selected_layer_types = list(
                    attr_value[args.layer_start : args.layer_start + args.num_layers]
                )
                break
    updated_keys = set(projection_info["updated_keys"])
    selection_notes = list(projection_info["selection_notes"])

    if isinstance(text_config_dict.get("full_attention_interval"), int) and selected_layer_types is not None:
        original_interval = text_config_dict["full_attention_interval"]
        projected_interval = project_full_attention_interval(selected_layer_types)
        if projected_interval is None:
            raise ValueError(
                "Cannot project the selected layer window into a local config for this model. "
                "The model derives local layer types from full_attention_interval, but the "
                f"window [{args.layer_start}, {args.layer_start + args.num_layers}) maps to "
                f"selected_layer_types={selected_layer_types}, which is not representable by a "
                "local periodic pattern."
            )
        text_config_dict["full_attention_interval"] = projected_interval
        updated_keys.add("full_attention_interval")
        selection_notes.append(
            f"full_attention_interval {original_interval} -> {projected_interval}"
        )

    mirror_projected_fields_to_ancestors(
        config_dict,
        text_config_path,
        text_config_dict,
        updated_keys,
    )

    temp_model_dir = tempfile.mkdtemp(prefix="sglang_decoder_layer_bench_")
    register_temp_path_for_cleanup(temp_model_dir)
    temp_config_path = os.path.join(temp_model_dir, "config.json")
    with open(temp_config_path, "w", encoding="utf-8") as fout:
        json.dump(config_dict, fout, indent=2, sort_keys=True)

    model_type = getattr(text_config_obj, "model_type", getattr(hf_config, "model_type", "unknown"))
    architectures = getattr(hf_config, "architectures", None) or []
    architecture = architectures[0] if architectures else type(hf_config).__name__
    special_token_ids = collect_special_token_ids(hf_config, text_config_obj)
    is_moe = detect_is_moe(text_config_obj, text_config_dict)
    prepared = PreparedModelConfig(
        architecture=architecture,
        model_type=model_type,
        text_config_path=text_config_path,
        original_num_layers=original_num_layers,
        effective_num_layers=args.num_layers,
        vocab_size=infer_vocab_size(hf_config, text_config_obj, text_config_dict),
        bos_token_id=infer_bos_token_id(hf_config, text_config_obj),
        special_token_ids=special_token_ids,
        is_moe=is_moe,
        temp_model_dir=temp_model_dir,
        temp_config_path=temp_config_path,
        override_notes="; ".join(
            [
                f"projected text config at {'.'.join(text_config_path) if text_config_path else '<root>'}",
                f"num_hidden_layers {original_num_layers} -> {args.num_layers}",
                *selection_notes,
            ]
        ),
        layer_start=args.layer_start,
        selected_layer_types=selected_layer_types,
        original_first_k_dense_replace=projection_info["original_first_k_dense_replace"],
        projected_first_k_dense_replace=projection_info["projected_first_k_dense_replace"],
        original_moe_layer_freq=projection_info["original_moe_layer_freq"],
        projected_moe_layer_freq=projection_info["projected_moe_layer_freq"],
        original_moe_layers=projection_info["original_moe_layers"],
        projected_moe_layers=projection_info["projected_moe_layers"],
        original_moe_layers_enum=projection_info["original_moe_layers_enum"],
        projected_moe_layers_enum=projection_info["projected_moe_layers_enum"],
        original_interleave_moe_layer_step=projection_info[
            "original_interleave_moe_layer_step"
        ],
        projected_interleave_moe_layer_step=projection_info[
            "projected_interleave_moe_layer_step"
        ],
        selection_notes=selection_notes,
    )
    prepared.single_layer_kind = infer_single_layer_kind(prepared)
    return prepared


def describe_selected_layers(model_cfg: PreparedModelConfig) -> str:
    end = model_cfg.layer_start + model_cfg.effective_num_layers
    parts = [
        f"layer_start={model_cfg.layer_start}",
        f"selected_layers=[{model_cfg.layer_start}, {end})",
        f"text_config_path={model_cfg.text_config_path_str}",
        f"total_layers={model_cfg.original_num_layers}",
    ]
    if model_cfg.selected_layer_types is not None:
        parts.append(f"selected_layer_types={model_cfg.selected_layer_types}")
    if model_cfg.original_first_k_dense_replace is not None:
        parts.append(
            "dense_prefix/moe_freq="
            f"({model_cfg.original_first_k_dense_replace}, {model_cfg.original_moe_layer_freq})"
            f" -> ({model_cfg.projected_first_k_dense_replace}, {model_cfg.projected_moe_layer_freq})"
        )
    if model_cfg.original_moe_layers is not None:
        parts.append(
            f"moe_layers={model_cfg.original_moe_layers} -> {model_cfg.projected_moe_layers}"
        )
    if model_cfg.original_moe_layers_enum is not None:
        parts.append(
            "moe_layers_enum="
            f"{model_cfg.original_moe_layers_enum} -> {model_cfg.projected_moe_layers_enum}"
        )
    if model_cfg.original_interleave_moe_layer_step is not None:
        parts.append(
            "interleave_moe_layer_step="
            f"{model_cfg.original_interleave_moe_layer_step} -> {model_cfg.projected_interleave_moe_layer_step}"
        )
    if model_cfg.single_layer_kind is not None:
        parts.append(f"single_layer_kind={model_cfg.single_layer_kind}")
    return ", ".join(parts)


def build_input_ids(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    bos_token_id: Optional[int],
    special_token_ids: Sequence[int],
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    special_ids = {token_id for token_id in special_token_ids if 0 <= token_id < vocab_size}
    inputs: list[list[int]] = []
    for _ in range(batch_size):
        row: list[int] = []
        for token_idx in range(seq_len):
            if token_idx == 0 and bos_token_id is not None and 0 <= bos_token_id < vocab_size:
                row.append(bos_token_id)
                continue
            for _ in range(32):
                candidate = rng.randrange(vocab_size)
                if candidate not in special_ids:
                    row.append(candidate)
                    break
            else:
                for candidate in range(vocab_size):
                    if candidate not in special_ids:
                        row.append(candidate)
                        break
                else:
                    raise ValueError(
                        "Unable to sample non-special token ids because special tokens cover the whole vocabulary."
                    )
        inputs.append(row)
    return inputs


def shard_batch_round_robin(
    full_batch: list[list[int]],
    dp_rank: int,
    dp_size: int,
) -> list[list[int]]:
    if dp_size <= 1:
        return full_batch
    return [tokens for idx, tokens in enumerate(full_batch) if idx % dp_size == dp_rank]


def native_dp_attention_enabled(args: argparse.Namespace) -> bool:
    return bool(args.enable_dp_attention and args.dp > 1)


def get_attn_dp_rank_for_tp_rank(server_args: ServerArgs, tp_rank: int) -> int:
    if not server_args.enable_dp_attention or server_args.dp_size <= 1:
        return 0
    attn_tp_size = (
        server_args.tp_size // server_args.dp_size // server_args.attn_cp_size
    )
    return tp_rank // (attn_tp_size * server_args.attn_cp_size)


def get_parallelism_ranks_for_tp_rank(
    server_args: ServerArgs, tp_rank: int
) -> tuple[int, int, int]:
    attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
    attn_tp_size = server_args.tp_size // attn_dp_size // server_args.attn_cp_size
    attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
    moe_dp_rank = tp_rank // (server_args.tp_size // server_args.moe_dp_size)
    moe_ep_rank = (
        tp_rank
        % (server_args.tp_size // server_args.moe_dp_size)
        // (server_args.tp_size // server_args.moe_dp_size // server_args.ep_size)
    )
    return attn_cp_rank, moe_dp_rank, moe_ep_rank


def get_native_dp_shard_sizes(
    input_ids: Sequence[Sequence[int]], dp_size: int
) -> list[int]:
    return [
        len(shard_batch_round_robin(list(input_ids), dp_rank, dp_size))
        for dp_rank in range(dp_size)
    ]


def get_local_num_reqs_budget(
    input_ids: Sequence[Sequence[int]], enable_dp_attention: bool, dp_size: int
) -> int:
    if not enable_dp_attention or dp_size <= 1:
        return len(input_ids)
    shard_sizes = get_native_dp_shard_sizes(input_ids, dp_size)
    return max(shard_sizes, default=0)


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _get_dist_reduce_device() -> torch.device:
    if not (dist.is_available() and dist.is_initialized()):
        return torch.device("cpu")
    backend = dist.get_backend()
    backend_name = backend if isinstance(backend, str) else str(backend)
    if "nccl" in backend_name.lower():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_reduce_max(value: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(
        value,
        dtype=torch.float64,
        device=_get_dist_reduce_device(),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def all_gather_object(obj: Any) -> list[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def init_external_launcher(args: argparse.Namespace) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(
            "external_launcher requires torchrun. WORLD_SIZE is missing or invalid."
        )
    if world_size != args.dp:
        raise ValueError(
            f"external_launcher expects WORLD_SIZE == dp, but got WORLD_SIZE={world_size}, dp={args.dp}."
        )
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))
    return rank, local_rank


def _describe_port_owner(port: int) -> str:
    process = find_process_using_port(port)
    if process is None:
        return ""
    try:
        cmdline = process.cmdline()
    except Exception:
        cmdline = []
    try:
        name = process.name()
    except Exception:
        name = "<unknown>"
    return f" pid={process.pid} name={name} cmdline={cmdline}"


def validate_external_launcher_local_resources(
    args: argparse.Namespace, local_launcher_rank: int
) -> None:
    if args.distributed_executor_backend != "external_launcher":
        return

    engine_port = args.base_port + local_launcher_rank * PORT_STRIDE
    nccl_port = engine_port + NCCL_PORT_OFFSET
    for port_name, port in (
        ("engine port", engine_port),
        ("nccl port", nccl_port),
    ):
        if not is_port_available(port):
            raise ValueError(
                "external_launcher requires a dedicated per-rank port block, "
                f"but {port_name} {port} for local rank {local_launcher_rank} "
                f"is already in use.{_describe_port_owner(port)}"
            )

    if args.device not in (None, "cuda"):
        return
    if not torch.cuda.is_available():
        raise ValueError(
            "external_launcher currently expects CUDA-visible devices, "
            f"but torch.cuda.is_available() is False for local rank {local_launcher_rank}."
        )

    visible_gpu_count = torch.cuda.device_count()
    base_gpu_id = local_launcher_rank * args.tp
    max_gpu_id = base_gpu_id + args.tp - 1
    if max_gpu_id >= visible_gpu_count:
        raise ValueError(
            "external_launcher assigns one TP slice per torchrun local rank. "
            f"Local rank {local_launcher_rank} needs visible GPU ids "
            f"[{base_gpu_id}, {max_gpu_id}], but only {visible_gpu_count} "
            f"visible GPU(s) are available "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})."
        )


def destroy_external_launcher() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sanitize_external_launcher_env() -> None:
    for env_var in TORCHRUN_ENV_VARS_TO_CLEAR:
        os.environ.pop(env_var, None)
    for env_var in list(os.environ):
        if env_var.startswith("TORCHELASTIC_"):
            os.environ.pop(env_var, None)


def warn_rank0(message: str) -> None:
    if get_rank() == 0:
        print(f"Warning: {message}", flush=True)


def resolve_a2a_backend(choice: str) -> A2ABackendSpec:
    mapping = {
        "naive": ("none", "auto"),
        "deepep_high_throughput": ("deepep", "normal"),
        "deepep_low_latency": ("deepep", "low_latency"),
        "mori": ("mori", "normal"),
        "flashinfer_all2allv": ("flashinfer", "auto"),
        "mooncake": ("mooncake", "auto"),
        "nixl": ("nixl", "auto"),
    }
    if choice not in mapping:
        raise ValueError(f"Unsupported all2all backend: {choice}")
    moe_a2a_backend, deepep_mode = mapping[choice]
    return A2ABackendSpec(
        user_choice=choice,
        moe_a2a_backend=moe_a2a_backend,
        deepep_mode=deepep_mode,
    )


def resolve_runtime_parallel_config(
    args: argparse.Namespace,
    model_cfg: PreparedModelConfig,
    backend_spec: A2ABackendSpec,
) -> RuntimeParallelConfig:
    notes: list[str] = []
    tp_size = args.tp
    if model_cfg.is_moe:
        ep_size = args.ep if args.ep is not None else tp_size
        moe_a2a_backend = backend_spec.moe_a2a_backend
        deepep_mode = backend_spec.deepep_mode
        if moe_a2a_backend != "none":
            if tp_size <= 1:
                raise ValueError(
                    f"all2all backend '{backend_spec.user_choice}' requires tp > 1 in current SGLang."
                )
            if ep_size != tp_size:
                notes.append(
                    f"forcing ep_size from {ep_size} to {tp_size} because backend '{moe_a2a_backend}' ties EP to TP in current SGLang"
                )
            ep_size = tp_size
    else:
        if args.ep not in (None, 1):
            notes.append(f"forcing ep_size from {args.ep} to 1 for dense model")
        if backend_spec.moe_a2a_backend != "none":
            notes.append(
                f"forcing moe_a2a_backend from '{backend_spec.moe_a2a_backend}' to 'none' because model is not detected as MoE"
            )
        ep_size = 1
        moe_a2a_backend = "none"
        deepep_mode = "auto"
    return RuntimeParallelConfig(
        tp_size=tp_size,
        ep_size=ep_size,
        moe_a2a_backend=moe_a2a_backend,
        deepep_mode=deepep_mode,
        notes=notes,
    )


def create_server_args(
    args: argparse.Namespace,
    model_cfg: PreparedModelConfig,
    runtime_cfg: RuntimeParallelConfig,
    local_launcher_rank: int,
    local_num_reqs: int,
) -> ServerArgs:
    if args.distributed_executor_backend == "ray":
        raise ValueError(
            "backend 'ray' is not supported by this benchmark script."
        )

    requested_page_size = (
        args.block_size
        if args.block_size is not None
        else get_default_page_size(args.model_path)
    )
    page_size = requested_page_size if requested_page_size is not None else 1
    context_len = args.seq_len + args.output_len
    per_worker_max_running_requests = max(1, local_num_reqs)
    # `max_running_requests` is interpreted by ModelRunner as the global
    # request budget across DP workers, then internally converted to a
    # per-DP-worker pool size. For native DP attention, `local_num_reqs` is
    # already the per-shard request budget, so pass the global equivalent here
    # to avoid dividing by `dp` twice and collapsing req_to_token_pool to 1.
    explicit_max_running_requests = max(
        1,
        per_worker_max_running_requests
        * (args.dp if native_dp_attention_enabled(args) else 1),
    )
    explicit_max_total_tokens = max(
        page_size,
        per_worker_max_running_requests * (context_len + page_size),
    )

    server_kwargs: dict[str, Any] = dict(
        # Keep the original model path so trust_remote_code models can still
        # resolve auxiliary Python modules from the real repo/cache. Feed the
        # projected config through ServerArgs' override-config path instead of
        # pretending the temp directory is the full model directory.
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        served_model_name=args.model_path,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        decrypted_config_file=model_cfg.temp_config_path,
        load_format=args.load_format,
        dtype=args.dtype,
        device=args.device,
        tp_size=runtime_cfg.tp_size,
        dp_size=args.dp,
        ep_size=runtime_cfg.ep_size,
        enable_dp_attention=args.enable_dp_attention,
        enable_dp_lm_head=args.enable_dp_lm_head,
        moe_a2a_backend=runtime_cfg.moe_a2a_backend,
        moe_runner_backend=args.moe_runner_backend,
        deepep_mode=runtime_cfg.deepep_mode,
        deepep_config=args.deepep_config,
        enable_layerwise_nvtx_marker=args.enable_layerwise_nvtx_marker,
        fp8_gemm_runner_backend=args.fp8_gemm_backend,
        context_length=context_len,
        random_seed=args.seed,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=explicit_max_running_requests,
        max_total_tokens=explicit_max_total_tokens,
        # Keep max_mamba_cache_size unset even for attention-only projected
        # windows. Hybrid models still construct HybridReqToTokenPool, and
        # forcing this to zero collapses the req pool size to zero before the
        # first extend().
        max_mamba_cache_size=None,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        skip_tokenizer_init=True,
        log_level=args.log_level,
        port=args.base_port + local_launcher_rank * PORT_STRIDE,
    )
    if args.enable_cuda_graph:
        server_kwargs["disable_cuda_graph"] = False
    if requested_page_size is not None:
        server_kwargs["page_size"] = requested_page_size
    if args.distributed_executor_backend == "external_launcher":
        server_kwargs["nccl_port"] = (
            args.base_port + local_launcher_rank * PORT_STRIDE + NCCL_PORT_OFFSET
        )
    return ServerArgs(**server_kwargs)


def patch_runtime_for_projected_layer_window() -> None:
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )

    if not getattr(
        ModelRunnerKVCacheMixin.profile_max_num_token,
        "_decoder_layer_bench_patch",
        False,
    ):
        original_profile = ModelRunnerKVCacheMixin.profile_max_num_token

        def patched_profile(self, pre_model_load_memory):
            try:
                return original_profile(self, pre_model_load_memory)
            except ZeroDivisionError:
                if self.server_args.max_total_tokens is not None:
                    return self.server_args.max_total_tokens
                raise

        patched_profile._decoder_layer_bench_patch = True
        ModelRunnerKVCacheMixin.profile_max_num_token = patched_profile

    if not getattr(
        HybridLinearKVPool.get_v_head_dim,
        "_decoder_layer_bench_patch",
        False,
    ):
        original_get_v_head_dim = HybridLinearKVPool.get_v_head_dim

        def patched_get_v_head_dim(self):
            if self.full_layer_nums == 0:
                return self.head_dim
            return original_get_v_head_dim(self)

        patched_get_v_head_dim._decoder_layer_bench_patch = True
        HybridLinearKVPool.get_v_head_dim = patched_get_v_head_dim


class DirectBenchRunner:
    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner

    def clear(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def extend(self, reqs):
        return bench_extend(reqs, self.model_runner)

    def decode(self, next_token_ids, batch):
        return bench_decode(next_token_ids, batch, self.model_runner)

    def cleanup(self, batch) -> None:
        pass

    def synchronize(self) -> None:
        torch.get_device_module(self.model_runner.device).synchronize()


def build_reqs_from_input_ids(
    input_ids: Sequence[Sequence[int]], output_len: int
) -> list[Req]:
    reqs: list[Req] = []
    for idx, token_ids in enumerate(input_ids):
        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=output_len,
            min_new_tokens=0,
            ignore_eos=True,
            stop_token_ids=[],
        )
        req = Req(
            rid=idx,
            origin_input_text="",
            origin_input_ids=list(token_ids),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)
    return reqs


def tp_all_reduce_max(value: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(
        value,
        dtype=torch.float64,
        device=_get_dist_reduce_device(),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _build_uniform_rank_router_logits(
    hidden_states: torch.Tensor,
    *,
    num_experts: int,
    routed_top_k: int,
    ep_size: int,
    experts_per_rank: int,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    logits = torch.full(
        (num_tokens, num_experts),
        -1e4,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    if num_tokens == 0:
        return logits

    token_idx = torch.arange(num_tokens, device=hidden_states.device, dtype=torch.int64)
    start_rank = token_idx.remainder(ep_size)
    local_base = (token_idx // ep_size).remainder(experts_per_rank)

    for slot in range(routed_top_k):
        rank = (start_rank + slot).remainder(ep_size)
        local_offset = (local_base + slot // ep_size).remainder(experts_per_rank)
        expert_id = rank * experts_per_rank + local_offset
        logits[token_idx, expert_id] = float(routed_top_k - slot)

    return logits


def maybe_patch_uniform_rank_moe_router(model_runner: ModelRunner, router_mode: str) -> int:
    if router_mode != "uniform_rank":
        return 0

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.glm4_moe import Glm4MoeSparseMoeBlock
    from sglang.srt.models.llama4 import Llama4MoE
    from sglang.srt.models.minimax_m2 import MiniMaxM2MoE
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    supported_moe_classes = (
        DeepseekV2MoE,
        Glm4MoeSparseMoeBlock,
        Qwen2MoeSparseMoeBlock,
        MiniMaxM2MoE,
        Llama4MoE,
    )

    def uniform_rank_forward_tensor(
        self,
        hidden_states,
        *args: Any,
        **kwargs: Any,
    ):
        del args, kwargs
        return _build_uniform_rank_router_logits(
            hidden_states,
            num_experts=self._bench_uniform_num_experts,
            routed_top_k=self._bench_uniform_routed_top_k,
            ep_size=self._bench_uniform_ep_size,
            experts_per_rank=self._bench_uniform_experts_per_rank,
        )

    def uniform_rank_forward_tuple(
        self,
        hidden_states,
        *args: Any,
        **kwargs: Any,
    ):
        del args, kwargs
        logits = _build_uniform_rank_router_logits(
            hidden_states,
            num_experts=self._bench_uniform_num_experts,
            routed_top_k=self._bench_uniform_routed_top_k,
            ep_size=self._bench_uniform_ep_size,
            experts_per_rank=self._bench_uniform_experts_per_rank,
        )
        return logits, None

    patched = 0
    with torch.no_grad():
        for module in model_runner.model.modules():
            if not isinstance(module, supported_moe_classes):
                continue

            if isinstance(module, Llama4MoE):
                route_module = module.router
                returns_tuple = True
                correction_bias = None
            else:
                route_module = module.gate
                returns_tuple = isinstance(
                    module, (Qwen2MoeSparseMoeBlock, MiniMaxM2MoE)
                )
                correction_bias = getattr(
                    module.gate, "e_score_correction_bias", None
                )
                if correction_bias is None:
                    correction_bias = getattr(module, "e_score_correction_bias", None)

            if getattr(route_module, "_bench_uniform_rank_router", False):
                continue

            if not hasattr(route_module, "weight"):
                raise ValueError(
                    f"uniform_rank router patch expects a weight-bearing routing module, got {type(route_module).__name__}."
                )

            num_experts = int(route_module.weight.shape[0])
            num_fused_shared_experts = int(
                getattr(module.topk.topk_config, "num_fused_shared_experts", 0)
            )
            routed_top_k = int(module.topk.topk_config.top_k) - num_fused_shared_experts
            ep_size = max(
                int(
                    getattr(
                        module.experts,
                        "moe_ep_size",
                        getattr(module, "moe_ep_size", getattr(module, "ep_size", 1)),
                    )
                ),
                1,
            )
            if num_experts % ep_size != 0:
                raise ValueError(
                    "uniform_rank router requires routed experts to divide evenly "
                    f"across EP ranks, got num_experts={num_experts}, ep_size={ep_size}."
                )
            if routed_top_k <= 0:
                raise ValueError(
                    f"uniform_rank router requires positive routed_top_k, got {routed_top_k}."
                )

            route_module._bench_uniform_num_experts = num_experts
            route_module._bench_uniform_routed_top_k = routed_top_k
            route_module._bench_uniform_ep_size = ep_size
            route_module._bench_uniform_experts_per_rank = num_experts // ep_size
            route_module.forward = MethodType(
                uniform_rank_forward_tuple if returns_tuple else uniform_rank_forward_tensor,
                route_module,
            )
            route_module._bench_uniform_rank_router = True

            if correction_bias is not None:
                correction_bias.zero_()

            if (
                module.topk.topk_config.use_grouped_topk
                and module.topk.topk_config.num_expert_group is not None
            ):
                # Keep the grouped-topk path for DeepSeek shared-expert handling,
                # but remove group pruning so round-robin logits are preserved.
                module.topk.topk_config.topk_group = (
                    module.topk.topk_config.num_expert_group
                )

            patched += 1

    if patched == 0:
        raise ValueError(
            "uniform_rank router mode only supports the requested MoE families "
            "(GLM-5, Llama-4 Maverick, Qwen3.5-397B-A17B, DeepSeek-V3.2-Exp, MiniMax-M2.5) "
            "in the current benchmark patch, but no supported MoE layers were found."
        )

    return patched


def load_direct_model_runner(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_router_mode: str,
) -> DirectBenchRunner:
    attn_cp_rank, moe_dp_rank, moe_ep_rank = get_parallelism_ranks_for_tp_rank(
        server_args, tp_rank
    )
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
    )
    patched_moe_layers = maybe_patch_uniform_rank_moe_router(
        model_runner, moe_router_mode
    )
    if tp_rank == 0 and patched_moe_layers > 0:
        print(
            f"Patched {patched_moe_layers} DeepSeek MoE layer(s) with router_mode={moe_router_mode}.",
            flush=True,
        )
    if server_args.tp_size > 1:
        dist.barrier()
    return DirectBenchRunner(model_runner)


def run_direct_iteration(
    model_runner: DirectBenchRunner,
    local_input_ids: Sequence[Sequence[int]],
    output_len: int,
) -> float:
    model_runner.clear()
    reqs = build_reqs_from_input_ids(local_input_ids, output_len)
    model_runner.synchronize()
    t0 = time.perf_counter()
    next_token_ids, _, batch = model_runner.extend(reqs)
    for _ in range(max(output_len - 1, 0)):
        next_token_ids, _ = model_runner.decode(next_token_ids, batch)
    model_runner.synchronize()
    latency = time.perf_counter() - t0
    model_runner.cleanup(batch)
    return tp_all_reduce_max(latency)


def direct_bench_worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_router_mode: str,
    local_input_ids: Sequence[Sequence[int]],
    output_len: int,
    num_warmup_iters: int,
    num_iters: int,
    enable_profile: bool,
    result_queue: mp.Queue,
) -> None:
    suppress_other_loggers()
    patch_runtime_for_projected_layer_window()
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_runner = load_direct_model_runner(
        server_args, port_args, gpu_id, tp_rank, moe_router_mode
    )

    for _ in range(num_warmup_iters):
        run_direct_iteration(model_runner, local_input_ids, output_len)

    profile_latency = None
    if enable_profile:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        profiler = start_profile(["CUDA_PROFILER"], rank_print=rank_print)
        profile_latency = run_direct_iteration(model_runner, local_input_ids, output_len)
        stop_profile(profiler, ["CUDA_PROFILER"], rank_print=rank_print)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    latencies = [
        run_direct_iteration(model_runner, local_input_ids, output_len)
        for _ in range(num_iters)
    ]

    if tp_rank == 0:
        result_queue.put(
            {
                "profile_latency": profile_latency,
                "latencies": latencies,
                "max_total_num_tokens": model_runner.model_runner.max_total_num_tokens,
                "page_size": model_runner.model_runner.server_args.page_size,
            }
        )

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "seq_len", "output_len", "num_layers", "tp", "dp"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}.")
    if args.num_iters <= 0:
        raise ValueError(f"num_iters must be > 0, got {args.num_iters}.")
    if args.num_warmup_iters < 0:
        raise ValueError(
            f"num_warmup_iters must be >= 0, got {args.num_warmup_iters}."
        )
    if args.layer_start < 0:
        raise ValueError(f"layer_start must be >= 0, got {args.layer_start}.")
    if args.ep is not None and args.ep <= 0:
        raise ValueError(f"ep must be > 0 when provided, got {args.ep}.")
    if (
        args.enable_dp_attention
        and args.distributed_executor_backend == "external_launcher"
    ):
        raise ValueError(
            "launcher-side DP and native --enable-dp-attention are different execution modes. "
            "Use one or the other, not both."
        )
    if native_dp_attention_enabled(args):
        if args.tp % args.dp != 0:
            raise ValueError(
                f"--enable-dp-attention requires tp to be divisible by dp, got tp={args.tp}, dp={args.dp}."
            )
    elif args.dp > 1 and args.distributed_executor_backend != "external_launcher":
        raise ValueError(
            "dp > 1 requires either --enable-dp-attention (native SGLang DP attention) "
            "or --distributed-executor-backend external_launcher (launcher-side batch sharding)."
        )
    if (
        args.distributed_executor_backend == "external_launcher"
        and args.batch_size < args.dp
    ):
        raise ValueError(
            f"external_launcher requires batch_size >= dp, got batch_size={args.batch_size}, dp={args.dp}."
        )
    if args.distributed_executor_backend == "uni":
        if args.tp != 1 or args.dp != 1:
            raise ValueError(
                "backend 'uni' only supports tp=1 and dp=1 in the current script."
            )
    if args.distributed_executor_backend == "ray":
        raise ValueError(
            "backend 'ray' is not supported by this script's local ModelRunner path in the current repository."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a single decoder layer or a local decoder layer window through SGLang's offline ModelRunner path."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--distributed-executor-backend",
        type=str,
        default="mp",
        choices=["mp", "ray", "uni", "external_launcher"],
    )
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument(
        "--enable-dp-attention",
        action="store_true",
        help="Use native SGLang DP attention inside one TP world. This is different from external_launcher batch sharding.",
    )
    parser.add_argument(
        "--enable-dp-lm-head",
        action="store_true",
        help="Enable DP LM head together with --enable-dp-attention.",
    )
    parser.add_argument(
        "--ep",
        "--expert-parallel-size",
        dest="ep",
        type=int,
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Maps to SGLang page_size.",
    )
    parser.add_argument("--load-format", type=str, default="dummy")
    parser.add_argument(
        "--all2all-backend",
        type=str,
        default="deepep_high_throughput",
        choices=[
            "naive",
            "deepep_high_throughput",
            "deepep_low_latency",
            "mori",
            "flashinfer_all2allv",
            "mooncake",
            "nixl",
        ],
    )
    parser.add_argument(
        "--deepep-config",
        type=str,
        default=None,
        help="DeepEP normal dispatch/combine config as JSON string or a path to a JSON file.",
    )
    parser.add_argument(
        "--moe-runner-backend",
        type=str,
        default="auto",
        choices=MOE_RUNNER_BACKEND_CHOICES,
    )
    parser.add_argument(
        "--fp8-gemm-backend",
        
        type=str,
        default="auto",
        choices=FP8_GEMM_RUNNER_BACKEND_CHOICES,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup-iters", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--enable-layerwise-nvtx-marker",
        action="store_true",
        help="Enable SGLang's built-in layerwise NVTX annotations for the model.",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Explicitly enable SGLang CUDA graph for decode. When omitted, keep SGLang's default behavior.",
    )
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--base-port", type=int, default=30000)
    parser.add_argument("--log-level", type=str, default="error")
    parser.add_argument(
        "--moe-router-mode",
        type=str,
        default="random",
        choices=["random", "uniform_rank"],
        help=(
            "Routing behavior for MoE benchmark. "
            "'random' keeps the existing dummy-initialized router; "
            "'uniform_rank' replaces DeepSeek MoE router logits with a "
            "deterministic round-robin pattern so tokens are balanced across EP ranks."
        ),
    )
    return parser


def print_config_summary(
    args: argparse.Namespace,
    launcher_batch_sizes: list[int],
    native_dp_shard_sizes: Optional[list[int]],
    runtime_cfg: RuntimeParallelConfig,
    server_args: ServerArgs,
) -> None:
    print(
        "Config: "
        f"batch_size={args.batch_size}, "
        f"launcher_batch_sizes={launcher_batch_sizes}, "
        f"native_dp_shard_sizes={native_dp_shard_sizes}, "
        f"seq_len={args.seq_len}, "
        f"output_len={args.output_len}, "
        f"num_layers={args.num_layers}, "
        f"layer_start={args.layer_start}, "
        f"tp={args.tp}, "
        f"dp={args.dp}, "
        f"enable_dp_attention={args.enable_dp_attention}, "
        f"ep_size={server_args.ep_size}, "
        f"all2all_backend={args.all2all_backend}, "
        f"moe_a2a_backend={server_args.moe_a2a_backend}, "
        f"moe_runner_backend={server_args.moe_runner_backend}, "
        f"deepep_mode={server_args.deepep_mode}, "
        f"deepep_config_set={bool(server_args.deepep_config)}, "
        f"enable_layerwise_nvtx_marker={server_args.enable_layerwise_nvtx_marker}, "
        f"fp8_gemm_backend={server_args.fp8_gemm_runner_backend}, "
        f"executor_backend={args.distributed_executor_backend}, "
        f"load_format={args.load_format}, "
        f"moe_router_mode={args.moe_router_mode}, "
        f"requested_block_size={args.block_size}, "
        f"resolved_page_size={server_args.page_size}, "
        f"nccl_port={server_args.nccl_port}, "
        f"cuda_graph_enabled={not server_args.disable_cuda_graph}, "
        f"piecewise_cuda_graph_enabled={not server_args.disable_piecewise_cuda_graph}, "
        f"dtype={args.dtype}, "
        f"device={args.device}, "
        f"mem_fraction_static={args.mem_fraction_static}",
        flush=True,
    )
    for note in runtime_cfg.notes:
        warn_rank0(note)


def run_direct_benchmark(
    args: argparse.Namespace,
    server_args: ServerArgs,
    local_launcher_rank: int,
    local_input_ids: Sequence[Sequence[int]],
) -> dict[str, Any]:
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    # CUDA/NCCL/DeepEP workers should use spawn. Fork can inherit partially
    # initialized CUDA runtime state from the parent and lead to undefined
    # failures deep inside communication kernels.
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    workers: list[mp.Process] = []

    base_gpu_id = (
        local_launcher_rank * args.tp
        if args.distributed_executor_backend == "external_launcher"
        else 0
    )

    for tp_rank in range(server_args.tp_size):
        worker_input_ids = [list(ids) for ids in local_input_ids]
        if server_args.enable_dp_attention and server_args.dp_size > 1:
            attn_dp_rank = get_attn_dp_rank_for_tp_rank(server_args, tp_rank)
            worker_input_ids = shard_batch_round_robin(
                worker_input_ids, attn_dp_rank, server_args.dp_size
            )
        with maybe_reindex_device_id(base_gpu_id + tp_rank) as gpu_id:
            proc = ctx.Process(
                target=direct_bench_worker,
                args=(
                    server_args,
                    port_args,
                    gpu_id,
                    tp_rank,
                    args.moe_router_mode,
                    worker_input_ids,
                    args.output_len,
                    args.num_warmup_iters,
                    args.num_iters,
                    args.profile,
                    result_queue,
                ),
            )
            proc.start()
            workers.append(proc)

    for proc in workers:
        proc.join()

    failed = [
        f"pid={proc.pid}, exitcode={proc.exitcode}"
        for proc in workers
        if proc.exitcode != 0
    ]
    if failed:
        raise RuntimeError(f"Local TP worker(s) failed: {failed}")

    try:
        return result_queue.get(timeout=1)
    except Exception as exc:
        raise RuntimeError("Benchmark worker did not return results.") from exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    launcher_rank = 0
    local_launcher_rank = 0
    external_launcher_initialized = False

    try:
        if args.distributed_executor_backend == "external_launcher":
            launcher_rank, local_launcher_rank = init_external_launcher(args)
            external_launcher_initialized = True
            sanitize_external_launcher_env()
            validate_external_launcher_local_resources(args, local_launcher_rank)

        model_cfg = prepare_model_config(args)
        backend_spec = resolve_a2a_backend(args.all2all_backend)
        runtime_cfg = resolve_runtime_parallel_config(args, model_cfg, backend_spec)

        full_batch = build_input_ids(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=model_cfg.vocab_size,
            bos_token_id=model_cfg.bos_token_id,
            special_token_ids=model_cfg.special_token_ids,
            seed=args.seed,
        )
        launcher_input_ids = shard_batch_round_robin(
            full_batch, get_rank(), get_world_size()
        )
        launcher_batch_sizes = [
            int(x) for x in all_gather_object(len(launcher_input_ids))
        ]
        native_dp_shard_sizes = (
            get_native_dp_shard_sizes(launcher_input_ids, args.dp)
            if native_dp_attention_enabled(args)
            else None
        )
        local_num_reqs = get_local_num_reqs_budget(
            launcher_input_ids,
            enable_dp_attention=native_dp_attention_enabled(args),
            dp_size=args.dp,
        )

        server_args = create_server_args(
            args,
            model_cfg,
            runtime_cfg,
            local_launcher_rank,
            local_num_reqs,
        )

        if launcher_rank == 0:
            print_config_summary(
                args,
                launcher_batch_sizes,
                native_dp_shard_sizes,
                runtime_cfg,
                server_args,
            )
            print(
                f"Layer selection: {describe_selected_layers(model_cfg)}",
                flush=True,
            )
            print(
                "Temp config: "
                f"text_config_path={model_cfg.text_config_path_str}, "
                f"temp_model_dir={model_cfg.temp_model_dir}, "
                f"temp_config_path={model_cfg.temp_config_path}",
                flush=True,
            )
            print(f"Override notes: {model_cfg.override_notes}", flush=True)

        if not model_cfg.is_moe and args.all2all_backend != "naive":
            warn_rank0(
                f"model is not detected as MoE; using moe_a2a_backend={server_args.moe_a2a_backend}, ep_size={server_args.ep_size}"
            )

        local_result = run_direct_benchmark(
            args,
            server_args,
            local_launcher_rank,
            launcher_input_ids,
        )
        profile_latency = local_result["profile_latency"]
        latencies = [all_reduce_max(float(x)) for x in local_result["latencies"]]

        if args.profile and profile_latency is not None:
            profile_latency = all_reduce_max(float(profile_latency))
            if launcher_rank == 0:
                print(f"Profile latency: {profile_latency * 1000.0:.3f} ms", flush=True)

        if launcher_rank == 0:
            prompt_tokens_per_iter = args.batch_size * args.seq_len
            decode_tokens_per_iter = args.batch_size * args.output_len
            print(
                "Benchmark: "
                f"mean_latency={statistics.mean(latencies) * 1000.0:.3f} ms, "
                f"min_latency={min(latencies) * 1000.0:.3f} ms, "
                f"max_latency={max(latencies) * 1000.0:.3f} ms, "
                f"prompt_tokens_per_iter={prompt_tokens_per_iter}, "
                f"decode_tokens_per_iter={decode_tokens_per_iter}",
                flush=True,
            )
    finally:
        if external_launcher_initialized:
            destroy_external_launcher()


if __name__ == "__main__":
    main()
