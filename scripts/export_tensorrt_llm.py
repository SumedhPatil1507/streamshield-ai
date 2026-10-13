"""Build StreamShield classifier engines and optional TensorRT-LLM engines.

The fixed-length toxicity classifier and image CNN are ONNX classifiers, so
they are compiled by ONNX Runtime's TensorRT Execution Provider. Paged KV
attention is an autoregressive LLM feature; it is only applied to an optional
TensorRT-LLM converted causal-LM checkpoint passed with ``--llm-checkpoint-dir``.

Examples (run in a CUDA/TensorRT environment)::

    python scripts/export_tensorrt_llm.py --output-dir models/trt
    python scripts/export_tensorrt_llm.py --output-dir models/trt \
        --llm-checkpoint-dir /models/guard-trtllm-checkpoint

The first invocation exports the repository's reference classifier weights if
the ONNX files are absent, then creates/warmups TensorRT EP engine caches. A
production deployment should point ``--text-onnx`` and ``--vision-onnx`` at
ONNX exports containing its trained weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_onnx_tensorrt import (  # noqa: E402
    export_text_model_to_onnx,
    export_vision_model_to_onnx,
)

LOG = logging.getLogger("streamshield.tensorrt_llm_export")


def _classifier_provider_options(
    cache_dir: Path,
    model_kind: str,
    max_batch_size: int,
    opt_batch_size: int,
    fp16: bool,
) -> Dict[str, object]:
    """Return explicit dynamic-shape profiles for the ORT TensorRT EP."""
    if model_kind == "text":
        min_shape = "input_ids:1x128,attention_mask:1x128"
        opt_shape = f"input_ids:{opt_batch_size}x128,attention_mask:{opt_batch_size}x128"
        max_shape = f"input_ids:{max_batch_size}x128,attention_mask:{max_batch_size}x128"
    else:
        min_shape = "pixel_values:1x3x224x224"
        opt_shape = f"pixel_values:{opt_batch_size}x3x224x224"
        max_shape = f"pixel_values:{max_batch_size}x3x224x224"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "device_id": 0,
        "trt_fp16_enable": fp16,
        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir.resolve()),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str((cache_dir / "timing.cache").resolve()),
        "trt_builder_optimization_level": 5,
        "trt_profile_min_shapes": min_shape,
        "trt_profile_opt_shapes": opt_shape,
        "trt_profile_max_shapes": max_shape,
    }


def build_classifier_engine(
    model_path: Path,
    cache_dir: Path,
    model_kind: str,
    max_batch_size: int,
    opt_batch_size: int,
    fp16: bool,
) -> Dict[str, object]:
    """Create and warm up a TensorRT EP session, populating its engine cache."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in available:
        raise RuntimeError(
            "This build of ONNX Runtime has no TensorrtExecutionProvider. "
            "Install a CUDA/TensorRT-enabled ONNX Runtime build on the target host."
        )

    provider_options = _classifier_provider_options(
        cache_dir, model_kind, max_batch_size, opt_batch_size, fp16
    )
    session = ort.InferenceSession(
        str(model_path),
        providers=[
            ("TensorrtExecutionProvider", provider_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    # Run the optimal shape once so the TensorRT EP builds/caches its engine.
    if model_kind == "text":
        batch = opt_batch_size
        inputs = {
            "input_ids": np.zeros((batch, 128), dtype=np.int64),
            "attention_mask": np.ones((batch, 128), dtype=np.int64),
        }
    else:
        inputs = {
            "pixel_values": np.zeros(
                (opt_batch_size, 3, 224, 224), dtype=np.float32
            )
        }
    session.run(None, inputs)
    return {
        "onnx": str(model_path.resolve()),
        "engine_cache": str(cache_dir.resolve()),
        "providers": session.get_providers(),
        "profile": {
            "min_batch": 1,
            "opt_batch": opt_batch_size,
            "max_batch": max_batch_size,
        },
    }


def build_llm_engine(
    checkpoint_dir: Path,
    output_dir: Path,
    max_batch_size: int,
    max_input_len: int,
    max_seq_len: int,
    max_num_tokens: int,
) -> List[str]:
    """Build a TensorRT-LLM engine with paged KV cache for a converted LLM."""
    builder = shutil.which("trtllm-build")
    if builder is None:
        raise RuntimeError("trtllm-build is not on PATH; install TensorRT-LLM in this environment.")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"TensorRT-LLM checkpoint directory not found: {checkpoint_dir}")
    command = [
        builder,
        "--checkpoint_dir", str(checkpoint_dir.resolve()),
        "--output_dir", str(output_dir.resolve()),
        "--kv_cache_type", "paged",
        "--use_paged_context_fmha", "enable",
        "--max_batch_size", str(max_batch_size),
        "--max_input_len", str(max_input_len),
        "--max_seq_len", str(max_seq_len),
        "--max_num_tokens", str(max_num_tokens),
        "--gemm_plugin", "auto",
        "--gpt_attention_plugin", "auto",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Building paged-KV TensorRT-LLM engine: %s", " ".join(command))
    subprocess.run(command, check=True)
    return command


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "tensorrt_llm")
    parser.add_argument("--text-onnx", type=Path, help="Trained fixed-length text classifier ONNX")
    parser.add_argument("--vision-onnx", type=Path, help="Trained vision classifier ONNX")
    parser.add_argument(
        "--llm-checkpoint-dir", type=Path,
        help="Optional TensorRT-LLM converted causal-LM checkpoint (not the repo classifier)",
    )
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--opt-batch-size", type=int, default=16)
    parser.add_argument("--max-input-len", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 TensorRT EP classifier engines")
    parser.add_argument(
        "--export-only", action="store_true",
        help="Export missing classifier ONNX files without requiring CUDA/TensorRT",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if not 1 <= args.opt_batch_size <= args.max_batch_size:
        parser.error("--opt-batch-size must be between 1 and --max-batch-size")
    if args.max_input_len <= 0 or args.max_seq_len < args.max_input_len:
        parser.error("Lengths must be positive and max-seq-len >= max-input-len")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    text_onnx = (args.text_onnx or output_dir / "text_toxicity.onnx").resolve()
    vision_onnx = (args.vision_onnx or output_dir / "image_classifier.onnx").resolve()

    if not text_onnx.exists():
        export_text_model_to_onnx(text_onnx)
    if not vision_onnx.exists():
        export_vision_model_to_onnx(vision_onnx)

    manifest: Dict[str, object] = {
        "classifier_backend": "onnxruntime_tensorrt_execution_provider",
        "classifiers": {},
        "llm_backend": None,
    }
    if not args.export_only:
        classifiers = manifest["classifiers"]
        assert isinstance(classifiers, dict)
        classifiers["text"] = build_classifier_engine(
            text_onnx, output_dir / "text_trt_cache", "text",
            args.max_batch_size, args.opt_batch_size, args.fp16,
        )
        classifiers["vision"] = build_classifier_engine(
            vision_onnx, output_dir / "vision_trt_cache", "vision",
            args.max_batch_size, args.opt_batch_size, args.fp16,
        )

    if args.llm_checkpoint_dir:
        command = build_llm_engine(
            args.llm_checkpoint_dir,
            output_dir / "llm_engine",
            args.max_batch_size,
            args.max_input_len,
            args.max_seq_len,
            args.max_num_tokens,
        )
        manifest["llm_backend"] = {
            "engine_dir": str((output_dir / "llm_engine").resolve()),
            "build_command": command,
            "kv_cache": "paged",
            "continuous_batching": (
                "inflight batching is handled by TensorRT-LLM's serving runtime; "
                "the engine build alone does not run a request scheduler"
            ),
        }

    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("Build manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
