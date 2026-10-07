"""
StreamShield AI — inference engine warmup.

Running a few dummy batches through the ONNX execution providers before
serving real traffic eliminates the JIT compilation latency that would
otherwise be incurred on the first real batch.

Set the environment variable ``STREAMSHIELD_OFFLINE=1`` (or any truthy value)
to skip outbound HTTP image fetches during warmup.  In offline mode a synthetic
black-pixel image is used instead, so the warmup still exercises the full ONNX
graph without requiring network access.  This is safe for air-gapped CI
environments and unit-test runners.
"""

from __future__ import annotations

import io
import os

import numpy as np
import structlog
from PIL import Image

from src.streaming.schemas import MultimodalPayload

from .inference import InferenceEngine

logger = structlog.get_logger(__name__)

# A publicly accessible 224×224 JPEG reachable from most environments.
_WARMUP_IMAGE_URL = "https://picsum.photos/224"

# Synthetic black JPEG used when STREAMSHIELD_OFFLINE is truthy.
_OFFLINE_PLACEHOLDER_URL = "https://cdn.streamshield.ai/img/warmup-placeholder.jpg"


def _is_offline() -> bool:
    """Return ``True`` when the ``STREAMSHIELD_OFFLINE`` env var is set."""
    return os.environ.get("STREAMSHIELD_OFFLINE", "").strip() not in ("", "0", "false", "False")


def _make_black_jpeg_bytes() -> bytes:
    """Return a minimal 224×224 black JPEG as raw bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


async def warmup(engine: InferenceEngine, n: int = 3) -> None:
    """Warm up *engine* by running *n* dummy batches of size 4.

    In **online mode** (default) each dummy payload uses ``_WARMUP_IMAGE_URL``
    so the image-fetch path is exercised end-to-end.

    In **offline mode** (``STREAMSHIELD_OFFLINE=1``) a synthetic black image is
    injected directly by monkey-patching the URL pool, avoiding all outbound
    HTTP.  The ONNX graph is still fully exercised.

    Parameters
    ----------
    engine:
        The ``InferenceEngine`` instance to warm up.
    n:
        Number of warmup passes (default 3).
    """
    offline = _is_offline()
    image_url = _WARMUP_IMAGE_URL

    if offline:
        logger.info("warmup_offline_mode", reason="STREAMSHIELD_OFFLINE is set")
        # In offline mode we patch load_image_bytes_async on model_utils so
        # the engine returns black-pixel bytes instead of making HTTP requests.
        from src.models import model_utils as _mu

        _original_fetch = _mu.load_image_bytes_async
        _black_bytes = _make_black_jpeg_bytes()

        async def _offline_fetch(url: str) -> bytes:  # noqa: ARG001
            return _black_bytes

        _mu.load_image_bytes_async = _offline_fetch  # type: ignore[assignment]

    logger.info("warmup_start", passes=n, batch_size=4, offline=offline)

    dummy_payloads = [
        MultimodalPayload(
            text_content="warmup dummy text for inference engine priming",
            image_url=image_url,
        )
        for _ in range(4)
    ]

    try:
        for i in range(n):
            results = await engine.run_batch(dummy_payloads)
            latency_ms = results[0].latency_ms if results else 0.0
            logger.info(
                "warmup_pass_complete",
                pass_number=i + 1,
                latency_ms=round(latency_ms, 2),
                offline=offline,
            )
    finally:
        if offline:
            # Restore the original fetch function unconditionally.
            _mu.load_image_bytes_async = _original_fetch  # type: ignore[assignment]

    logger.info("warmup_done", passes=n)
