"""
StreamShield AI — ONNX stub model exporter.

Generates two minimal but valid ONNX graphs that match the exact input/output
shapes expected by ``src/models/inference.py``.  These stubs are for local
development and CI; they should be replaced with real fine-tuned weights
before production deployment.

Usage
-----
Run from the workspace root::

    python scripts/export_onnx_stubs.py

Output
------
* ``models/text_toxicity.onnx``
* ``models/image_classifier.onnx``
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = WORKSPACE_ROOT / "models"


def _make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    """Wrap a numpy array as an ONNX initializer tensor."""
    return numpy_helper.from_array(array, name=name)


# ---------------------------------------------------------------------------
# Text toxicity stub
# ---------------------------------------------------------------------------
# Input : input_ids       int64  (batch, 128)
#         attention_mask  int64  (batch, 128)
# Output: logits          float32 (batch, 2)
#         embeddings      float32 (batch, 128)
#
# Graph: Cast(input_ids → float32) → Gemm → logits
#                                   → Gemm → embeddings
# ---------------------------------------------------------------------------


def build_text_stub() -> onnx.ModelProto:
    """Build the text toxicity ONNX stub graph."""

    rng = np.random.default_rng(42)

    # Weight matrices — small random values, transB=1 convention:
    #   W_logits     shape (2, 128)  → Gemm(A=(batch,128), B=(2,128), transB=1) → (batch,2)
    #   W_embeddings shape (128,128) → Gemm(A=(batch,128), B=(128,128),transB=1) → (batch,128)
    w_logits = rng.standard_normal((2, 128)).astype(np.float32) * 0.01
    b_logits = np.zeros(2, dtype=np.float32)
    w_emb = rng.standard_normal((128, 128)).astype(np.float32) * 0.01
    b_emb = np.zeros(128, dtype=np.float32)

    # Graph inputs
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", 128])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", 128]
    )

    # Graph outputs
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])
    embeddings_out = helper.make_tensor_value_info(
        "embeddings", TensorProto.FLOAT, ["batch", 128]
    )

    # Cast input_ids to float32 (attention_mask is unused by the stub but still
    # listed as an input so the interface matches the real model).
    cast_node = helper.make_node(
        "Cast",
        inputs=["input_ids"],
        outputs=["input_ids_float"],
        to=TensorProto.FLOAT,
    )

    # Gemm: logits
    gemm_logits = helper.make_node(
        "Gemm",
        inputs=["input_ids_float", "w_logits", "b_logits"],
        outputs=["logits"],
        transB=1,
    )

    # Gemm: embeddings
    gemm_emb = helper.make_node(
        "Gemm",
        inputs=["input_ids_float", "w_emb", "b_emb"],
        outputs=["embeddings"],
        transB=1,
    )

    initializers = [
        _make_initializer("w_logits", w_logits),
        _make_initializer("b_logits", b_logits),
        _make_initializer("w_emb", w_emb),
        _make_initializer("b_emb", b_emb),
    ]

    graph = helper.make_graph(
        nodes=[cast_node, gemm_logits, gemm_emb],
        name="text_toxicity_stub",
        inputs=[input_ids, attention_mask],
        outputs=[logits_out, embeddings_out],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Image classifier stub
# ---------------------------------------------------------------------------
# Input : pixel_values  float32 (batch, 3, 224, 224)
# Output: logits        float32 (batch, 2)
#         embeddings    float32 (batch, 128)
#
# Graph: Reshape(batch,3,224,224)→(batch,150528)
#        → Gemm → logits
#        → Gemm → embeddings
# ---------------------------------------------------------------------------


def build_image_stub() -> onnx.ModelProto:
    """Build the image classifier ONNX stub graph."""

    rng = np.random.default_rng(99)

    # transB=0 convention (no transpose): W columns are output features.
    #   W_logits     shape (150528, 2)
    #   W_embeddings shape (150528, 128)
    w_logits = rng.standard_normal((150528, 2)).astype(np.float32) * 0.001
    b_logits = np.zeros(2, dtype=np.float32)
    w_emb = rng.standard_normal((150528, 128)).astype(np.float32) * 0.001
    b_emb = np.zeros(128, dtype=np.float32)

    # Constant shape tensor for Reshape: [-1, 150528]
    reshape_shape = np.array([-1, 150528], dtype=np.int64)

    # Graph inputs / outputs
    pixel_values = helper.make_tensor_value_info(
        "pixel_values", TensorProto.FLOAT, ["batch", 3, 224, 224]
    )
    logits_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])
    embeddings_out = helper.make_tensor_value_info(
        "embeddings", TensorProto.FLOAT, ["batch", 128]
    )

    # Constant node for the reshape target shape
    shape_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reshape_shape"],
        value=helper.make_tensor(
            name="reshape_shape_val",
            data_type=TensorProto.INT64,
            dims=[2],
            vals=reshape_shape.tolist(),
        ),
    )

    # Reshape to (batch, 150528)
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["pixel_values", "reshape_shape"],
        outputs=["flat_pixels"],
    )

    # Gemm: logits — transB=0, so W is (150528, 2)
    gemm_logits = helper.make_node(
        "Gemm",
        inputs=["flat_pixels", "w_logits", "b_logits"],
        outputs=["logits"],
        transB=0,
    )

    # Gemm: embeddings — transB=0, so W is (150528, 128)
    gemm_emb = helper.make_node(
        "Gemm",
        inputs=["flat_pixels", "w_emb", "b_emb"],
        outputs=["embeddings"],
        transB=0,
    )

    initializers = [
        _make_initializer("w_logits", w_logits),
        _make_initializer("b_logits", b_logits),
        _make_initializer("w_emb", w_emb),
        _make_initializer("b_emb", b_emb),
    ]

    graph = helper.make_graph(
        nodes=[shape_const, reshape_node, gemm_logits, gemm_emb],
        name="image_classifier_stub",
        inputs=[pixel_values],
        outputs=[logits_out, embeddings_out],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Export both ONNX stub models to ``models/``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    text_path = MODELS_DIR / "text_toxicity.onnx"
    image_path = MODELS_DIR / "image_classifier.onnx"

    print("Building text toxicity stub…")
    text_model = build_text_stub()
    onnx.save(text_model, str(text_path))
    print(f"  ✓ Saved {text_path}")

    print("Building image classifier stub…")
    image_model = build_image_stub()
    onnx.save(image_model, str(image_path))
    print(f"  ✓ Saved {image_path}")

    print("All stubs exported successfully.")


if __name__ == "__main__":
    main()
