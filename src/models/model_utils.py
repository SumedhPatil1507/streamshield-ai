"""
StreamShield AI — model utilities.

Shared helpers for ONNX session management, numerical operations, and async
HTTP image loading. All ONNX sessions are cached in a module-level registry so
the same model file is never loaded twice within a process.
"""

from __future__ import annotations

import os
from typing import Dict

import httpx
import numpy as np
import onnxruntime as ort
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level session cache
# ---------------------------------------------------------------------------
_session_cache: Dict[str, ort.InferenceSession] = {}


def get_onnx_session(model_path: str) -> ort.InferenceSession:
    """Load (or return cached) an ONNX ``InferenceSession`` for *model_path*.

    Provider priority is ``CUDAExecutionProvider`` → ``CPUExecutionProvider``.
    ``IntraOpNumThreads`` is set to ``os.cpu_count()`` and graph optimisation
    is set to ``ORT_ENABLE_ALL`` for maximum throughput.

    Parameters
    ----------
    model_path:
        Filesystem path to the ``.onnx`` model file.

    Returns
    -------
    ort.InferenceSession
        A fully initialised, cached ONNX runtime session.
    """
    if model_path in _session_cache:
        return _session_cache[model_path]

    desired_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Filter to only providers that are actually available in this build.
    available = ort.get_available_providers()
    providers = [p for p in desired_providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = os.cpu_count() or 1
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=session_opts, providers=providers)

    selected_provider = session.get_providers()[0]
    logger.info(
        "onnx_session_loaded",
        model_path=model_path,
        provider=selected_provider,
        all_providers=session.get_providers(),
    )

    _session_cache[model_path] = session
    return session


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid.

    For positive values uses ``1 / (1 + exp(-x))``.
    For negative values uses the equivalent ``exp(x) / (1 + exp(x))`` form
    to prevent overflow.  Each branch is computed independently so numpy
    never evaluates exp on out-of-range values for the path not taken.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    # Positive branch: safe because exp(-x) stays in [0, 1] when x >= 0.
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    # Negative branch: exp(x) stays in (0, 1) when x < 0.
    exp_x = np.exp(x[neg])
    result[neg] = exp_x / (1.0 + exp_x)
    return result.astype(np.float32)


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2-normalise *x* along its last axis.

    An epsilon of 1e-12 prevents division by zero for zero vectors.

    Parameters
    ----------
    x:
        Array of arbitrary shape; normalisation is applied along ``axis=-1``.

    Returns
    -------
    np.ndarray
        Array of the same shape and dtype as *x* with unit L2 norm on the last
        axis.
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


# ---------------------------------------------------------------------------
# Async image fetching
# ---------------------------------------------------------------------------

async def load_image_bytes_async(url: str) -> bytes:
    """Fetch raw image bytes from *url* asynchronously.

    Uses ``httpx.AsyncClient`` with a 5-second timeout. Raises
    ``httpx.HTTPStatusError`` for non-2xx responses.

    Parameters
    ----------
    url:
        HTTP(S) URL of the image to download.

    Returns
    -------
    bytes
        Raw image bytes suitable for ``PIL.Image.open``.

    Raises
    ------
    httpx.HTTPStatusError
        When the server returns a non-2xx status code.
    httpx.TimeoutException
        When the request times out.
    """
    timeout = httpx.Timeout(5.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
