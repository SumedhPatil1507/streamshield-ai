"""
StreamShield AI — multimodal inference module.

Implements a production-grade ONNX-based inference engine that scores text and
image content for toxicity concurrently (via ``asyncio.gather``) and returns
combined moderation predictions in under 50 ms per batch.

Design notes
------------
* ONNX sessions are loaded once and cached by ``model_utils.get_onnx_session``.
* CPU-bound ONNX calls are offloaded to a thread-pool executor so they never
  block the asyncio event loop.
* Image fetches are performed concurrently; per-item fetch failures are
  gracefully handled by substituting a zero vector.
"""

from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import structlog
from PIL import Image

from src.streaming.config import settings
from src.streaming.schemas import MultimodalPayload

from .model_utils import (
    get_onnx_session,
    load_image_bytes_async,
    normalize_l2,
    sigmoid,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class ModerationResult:
    """Per-item result returned by the inference engine.

    Attributes
    ----------
    payload_id:
        Unique identifier copied from the source ``MultimodalPayload``.
    label:
        ``"Toxic"`` or ``"Non-Toxic"``.
    confidence:
        Scalar in ``[0, 1]`` — the maximum of *text_confidence* and
        *image_confidence*.
    text_confidence:
        Sigmoid probability of the toxic class from the text model.
    image_confidence:
        Sigmoid probability of the toxic class from the image model.
    text_embedding:
        128-dimensional L2-normalised text feature vector.
    image_embedding:
        128-dimensional L2-normalised image feature vector.
    latency_ms:
        Wall-clock time for the entire batch that contained this item.
    """

    payload_id: str
    label: str
    confidence: float
    text_confidence: float
    image_confidence: float
    text_embedding: List[float]
    image_embedding: List[float]
    latency_ms: float


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tokenize_batch(texts: List[str]) -> dict:
    """Tokenise a list of texts with the HuggingFace tokenizer.

    Loads ``AutoTokenizer`` from ``settings.TEXT_MODEL_PATH`` on first call.
    Tokens are padded/truncated to 128 subwords.

    Parameters
    ----------
    texts:
        List of raw text strings to tokenise.

    Returns
    -------
    dict
        Keys ``input_ids`` and ``attention_mask``, each an ``np.int64`` array
        of shape ``(len(texts), 128)``.
    """
    tokenizer = _get_tokenizer()
    encoded = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }


def preprocess_images_batch(image_bytes_list: List[bytes]) -> np.ndarray:
    """Decode and normalise a list of raw image bytes into a model-ready tensor.

    Parameters
    ----------
    image_bytes_list:
        List of raw JPEG/PNG bytes for each image in the batch.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(N, 3, 224, 224)`` normalised with ImageNet
        statistics (channels-first).
    """
    arrays: List[np.ndarray] = []
    for raw in image_bytes_list:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD  # normalise
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        arrays.append(arr)
    return np.stack(arrays, axis=0)  # (N, 3, 224, 224)


# ---------------------------------------------------------------------------
# Lazy tokenizer loader
# ---------------------------------------------------------------------------

_tokenizer: Optional[object] = None


def _get_tokenizer():
    """Return a cached ``AutoTokenizer`` instance."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer  # deferred import for speed

        logger.info("loading_tokenizer", model_path=settings.TEXT_MODEL_PATH)
        _tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_MODEL_PATH)
    return _tokenizer


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Multimodal ONNX inference engine.

    Text and image inference run **concurrently** via ``asyncio.gather``.
    ONNX sessions are initialised eagerly in ``__init__`` so the first batch
    is not penalised by model loading.

    Parameters
    ----------
    text_model_path:
        Path to the text toxicity ONNX model.
    image_model_path:
        Path to the image classifier ONNX model.
    """

    def __init__(self, text_model_path: str, image_model_path: str) -> None:
        self._text_model_path = text_model_path
        self._image_model_path = image_model_path
        # Eagerly load sessions so we fail fast on missing/corrupt models.
        self._text_session = get_onnx_session(text_model_path)
        self._image_session = get_onnx_session(image_model_path)
        logger.info(
            "inference_engine_ready",
            text_model=text_model_path,
            image_model=image_model_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_batch(
        self, batch: List[MultimodalPayload]
    ) -> List[ModerationResult]:
        """Score a batch of multimodal payloads concurrently.

        Text and image inference pipelines run in parallel via
        ``asyncio.gather``. Results are combined per-item and returned as a
        list of ``ModerationResult`` objects.

        Parameters
        ----------
        batch:
            List of ``MultimodalPayload`` items to score.

        Returns
        -------
        list[ModerationResult]
            One result per input item, in the same order.
        """
        start = time.perf_counter()

        texts = [item.text_content for item in batch]
        image_urls = [item.image_url for item in batch]

        # Run text and image inference concurrently.
        (text_logits, text_embeddings), (image_logits, image_embeddings) = (
            await asyncio.gather(
                self._run_text_inference(texts),
                self._run_image_inference(image_urls),
            )
        )

        results: List[ModerationResult] = []
        for i, item in enumerate(batch):
            text_prob: float = float(sigmoid(text_logits[i : i + 1])[0, 1])
            image_prob: float = float(sigmoid(image_logits[i : i + 1])[0, 1])
            confidence: float = max(text_prob, image_prob)
            label: str = "Toxic" if confidence >= 0.5 else "Non-Toxic"

            latency_ms = (time.perf_counter() - start) * 1000.0

            results.append(
                ModerationResult(
                    payload_id=item.payload_id,
                    label=label,
                    confidence=confidence,
                    text_confidence=text_prob,
                    image_confidence=image_prob,
                    text_embedding=normalize_l2(text_embeddings[i]).tolist(),
                    image_embedding=normalize_l2(image_embeddings[i]).tolist(),
                    latency_ms=latency_ms,
                )
            )

        total_latency_ms = (time.perf_counter() - start) * 1000.0
        log = logger.bind(batch_size=len(batch), latency_ms=round(total_latency_ms, 2))
        if total_latency_ms > 50.0:
            log.warning("batch_latency_exceeded_50ms")
        else:
            log.info("batch_inference_complete")

        return results

    # ------------------------------------------------------------------
    # Private inference helpers
    # ------------------------------------------------------------------

    async def _run_text_inference(
        self, texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenise *texts* and run the text ONNX session in an executor.

        Parameters
        ----------
        texts:
            Raw text strings for the batch.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(logits, embeddings)`` with shapes ``(N, 2)`` and ``(N, 128)``.
        """
        inputs = tokenize_batch(texts)
        session = self._text_session

        def _run() -> Tuple[np.ndarray, np.ndarray]:
            outputs = session.run(
                ["logits", "embeddings"],
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )
            return outputs[0], outputs[1]

        loop = asyncio.get_event_loop()
        logits, embeddings = await loop.run_in_executor(None, _run)
        return logits, embeddings

    async def _run_image_inference(
        self, image_urls: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch images concurrently, preprocess, and run the image ONNX session.

        Per-item HTTP errors are handled gracefully: failed images are replaced
        with a black (zero) image so the batch is never aborted.

        Parameters
        ----------
        image_urls:
            List of HTTP(S) image URLs.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(logits, embeddings)`` with shapes ``(N, 2)`` and ``(N, 128)``.
        """
        # Fetch all images concurrently; replace failures with zero tensors.
        fetch_tasks = [_safe_fetch(url) for url in image_urls]
        image_bytes_list: List[Optional[bytes]] = await asyncio.gather(*fetch_tasks)

        # Build pixel tensor, using a black image for any failed fetch.
        processed: List[np.ndarray] = []
        for i, raw in enumerate(image_bytes_list):
            if raw is None:
                # Zero-filled placeholder: (3, 224, 224)
                processed.append(np.zeros((3, 224, 224), dtype=np.float32))
            else:
                processed.append(
                    preprocess_images_batch([raw])[0]  # shape (3, 224, 224)
                )
        pixel_values = np.stack(processed, axis=0).astype(np.float32)  # (N, 3, 224, 224)

        session = self._image_session

        def _run() -> Tuple[np.ndarray, np.ndarray]:
            outputs = session.run(
                ["logits", "embeddings"],
                {"pixel_values": pixel_values},
            )
            return outputs[0], outputs[1]

        loop = asyncio.get_event_loop()
        logits, embeddings = await loop.run_in_executor(None, _run)
        return logits, embeddings


# ---------------------------------------------------------------------------
# Async safe image fetch helper
# ---------------------------------------------------------------------------


async def _safe_fetch(url: str) -> Optional[bytes]:
    """Fetch *url*, returning ``None`` on any HTTP or timeout error."""
    try:
        return await load_image_bytes_async(url)
    except (Exception,) as exc:  # noqa: BLE001
        logger.warning("image_fetch_failed", url=url, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Return the module-level ``InferenceEngine`` singleton.

    The engine is lazily initialised on the first call using paths from
    ``settings``.
    """
    global _engine
    if _engine is None:
        _engine = InferenceEngine(
            text_model_path=settings.TEXT_ONNX_PATH,
            image_model_path=settings.IMAGE_MODEL_PATH,
        )
    return _engine


async def run_batch(batch: List[MultimodalPayload]) -> List[ModerationResult]:
    """Top-level coroutine: score *batch* using the singleton engine.

    Parameters
    ----------
    batch:
        List of ``MultimodalPayload`` items to score.

    Returns
    -------
    list[ModerationResult]
        One ``ModerationResult`` per input item.
    """
    return await get_engine().run_batch(batch)
