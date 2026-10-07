"""
StreamShield AI — inference engine warmup.

Running a few dummy batches through the ONNX execution providers before
serving real traffic eliminates the JIT compilation latency that would
otherwise be incurred on the first real batch.
"""

from __future__ import annotations

import structlog

from src.streaming.schemas import MultimodalPayload

from .inference import InferenceEngine

logger = structlog.get_logger(__name__)

# A publicly accessible 224×224 JPEG that should be reachable from most
# environments without authentication.
_WARMUP_IMAGE_URL = "https://picsum.photos/224"


async def warmup(engine: InferenceEngine, n: int = 3) -> None:
    """Warm up *engine* by running *n* dummy batches of size 4.

    Each batch uses synthetic ``MultimodalPayload`` objects with a harmless
    text string and ``_WARMUP_IMAGE_URL`` as the image source.  Latency for
    each warmup pass is logged so operators can verify that the ONNX execution
    providers have fully initialised before production traffic is admitted.

    Parameters
    ----------
    engine:
        The ``InferenceEngine`` instance to warm up.
    n:
        Number of warmup passes (default 3).
    """
    logger.info("warmup_start", passes=n, batch_size=4)

    dummy_payloads = [
        MultimodalPayload(
            text_content="warmup dummy text for inference engine priming",
            image_url=_WARMUP_IMAGE_URL,
        )
        for _ in range(4)
    ]

    for i in range(n):
        results = await engine.run_batch(dummy_payloads)
        latency_ms = results[0].latency_ms if results else 0.0
        logger.info(
            "warmup_pass_complete",
            pass_number=i + 1,
            latency_ms=round(latency_ms, 2),
        )

    logger.info("warmup_done", passes=n)
