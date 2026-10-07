"""Inference pipeline for the StreamShield AI streaming system.

Provides ``run_batch``, which accepts a list of ``MultimodalPayload`` objects,
runs simulated GPU-latency inference concurrently via ``asyncio.gather``, and
returns a corresponding list of ``InferenceResult`` objects.
"""

import asyncio
import random
import time
from datetime import datetime, timezone
from typing import List

import structlog

from .schemas import InferenceResult, MultimodalPayload

logger = structlog.get_logger(__name__)

_LABELS: List[str] = ["safe", "flagged", "review"]


async def _infer_single(payload: MultimodalPayload) -> InferenceResult:
    """Run inference on a single payload.

    Simulates GPU latency with a short async sleep, then assigns a random
    label and confidence score.

    Parameters
    ----------
    payload:
        The multimodal payload to classify.

    Returns
    -------
    InferenceResult
        The classification result for the payload.
    """
    # Simulate GPU latency (5–20 ms) without blocking the event loop.
    await asyncio.sleep(random.uniform(0.005, 0.02))

    return InferenceResult(
        payload_id=payload.payload_id,
        label=random.choice(_LABELS),
        confidence=random.uniform(0.7, 1.0),
        processed_at=datetime.now(timezone.utc),
    )


async def run_batch(batch: List[MultimodalPayload]) -> List[InferenceResult]:
    """Run inference on a micro-batch of payloads concurrently.

    All payloads in *batch* are inferred in parallel using ``asyncio.gather``.
    Elapsed wall-clock time and batch size are logged after completion.

    Parameters
    ----------
    batch:
        List of ``MultimodalPayload`` instances to process.

    Returns
    -------
    list[InferenceResult]
        Inference results in the same order as the input batch.
    """
    start = time.monotonic()
    results: List[InferenceResult] = await asyncio.gather(
        *[_infer_single(p) for p in batch]
    )
    elapsed_ms = (time.monotonic() - start) * 1000.0

    logger.info(
        "inference_batch_complete",
        batch_size=len(batch),
        elapsed_ms=round(elapsed_ms, 2),
    )
    return list(results)
