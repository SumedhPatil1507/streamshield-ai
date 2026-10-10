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
    image_fetch_failed:
        ``True`` when the image could not be fetched and a zero-vector
        placeholder was used instead.  Downstream consumers can use this
        flag to down-weight image confidence or trigger a retry.
    """

    payload_id: str
    label: str
    confidence: float
    text_confidence: float
    image_confidence: float
    text_embedding: List[float]
    image_embedding: List[float]
    latency_ms: float
    image_fetch_failed: bool = False


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


def decode_image_to_chw01(raw: bytes) -> np.ndarray:
    """Decode raw image bytes to a ``(3, 224, 224)`` float32 CHW array in ``[0, 1]``.

    This is the *raw* (un-normalised) decode step so that normalisation can be
    executed either on the CPU (NumPy) or on the GPU inside
    :class:`PinnedImageBatchBuffer`'s dedicated CUDA stream.
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (224, 224, 3)
    return arr.transpose(2, 0, 1)  # HWC → CHW


def normalize_chw01_batch(batch: np.ndarray) -> np.ndarray:
    """ImageNet-normalise a ``(N, 3, 224, 224)`` float32 array in ``[0, 1]`` (CPU)."""
    mean = _IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std = _IMAGENET_STD.reshape(1, 3, 1, 1)
    return ((batch - mean) / std).astype(np.float32)


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
    decoded = np.stack([decode_image_to_chw01(raw) for raw in image_bytes_list], axis=0)
    return normalize_chw01_batch(decoded)


# ---------------------------------------------------------------------------
# Pinned (page-locked) host memory staging + asynchronous DMA CUDA streams
# ---------------------------------------------------------------------------


class PinnedImageBatchBuffer:
    """Pre-allocated page-locked host buffer and dedicated CUDA stream.

    HPC rationale
    -------------
    * ``torch.empty(..., pin_memory=True)`` allocates **page-locked** host RAM.
      Only pinned pages can be targeted by the GPU NIC's direct memory access
      (DMA); pageable memory must first be staged through a driver-internal
      pinned bounce buffer, serialising the copy.
    * ``torch.cuda.Stream()`` gives this buffer a **dedicated copy/compute
      stream**. The host→device transfer is issued with ``non_blocking=True``
      (legal only for pinned sources) so it overlaps with kernels already in
      flight instead of serialising on the default stream.
    * Normalisation (ImageNet mean/std) is fused into the same stream so the
      transfer and the scale/shift execute back-to-back before the single
      ``stream.synchronize()`` at the ONNX session boundary.

    On CPU-only builds (no CUDA accelerator) ``torch`` is imported lazily and
    the class transparently degrades to a NumPy normalisation path with the
    same numeric output — no CUDA imports, no pinning attempts.
    """

    def __init__(self, max_batch_size: int = 64, height: int = 224, width: int = 224) -> None:
        self.max_batch_size = max_batch_size
        self.height = height
        self.width = width
        self.cuda_available = False
        self.stream = None
        self.host_buffer = None
        self.device_buffer = None
        self._torch = None

        try:
            import torch  # lazy: keeps CPU-only startup fast

            self._torch = torch
            if torch.cuda.is_available():
                self.cuda_available = True
                # Dedicated CUDA stream for non-blocking H2D DMA transfers.
                self.stream = torch.cuda.Stream()
                # Page-locked (pinned) host staging buffer.
                self.host_buffer = torch.empty(
                    (max_batch_size, 3, height, width),
                    dtype=torch.float32,
                    pin_memory=True,
                )
                # Pre-allocated destination in device VRAM (reused every batch).
                self.device_buffer = torch.empty(
                    (max_batch_size, 3, height, width),
                    dtype=torch.float32,
                    device="cuda",
                )
                self._mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
                self._std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device="cuda").view(1, 3, 1, 1)
        except Exception:  # pragma: no cover - torch missing / driver error
            self.cuda_available = False

        logger.info(
            "pinned_image_buffer_ready",
            cuda_available=self.cuda_available,
            max_batch_size=max_batch_size,
            mode="cuda_pinned_dma" if self.cuda_available else "cpu_numpy_fallback",
        )

    @property
    def pinned_memory_used(self) -> bool:
        """``True`` when page-locked host memory + async DMA is active."""
        return self.cuda_available

    def normalize_batch(self, chw01_batch: np.ndarray) -> np.ndarray:
        """ImageNet-normalise ``(N, 3, 224, 224)`` float32 data in ``[0, 1]``.

        GPU path: pageable→pinned copy, **non-blocking pinned→VRAM DMA** on the
        dedicated stream, fused normalisation kernel, single stream sync.
        CPU path: vectorised NumPy normalisation (identical numerics).
        """
        if chw01_batch.ndim != 4 or chw01_batch.shape[1:] != (3, self.height, self.width):
            raise ValueError(f"Expected (N, 3, {self.height}, {self.width}) batch, got {chw01_batch.shape}")
        n = chw01_batch.shape[0]
        if n > self.max_batch_size:
            raise ValueError(f"Batch size {n} exceeds pinned buffer capacity {self.max_batch_size}")

        if not self.cuda_available:
            return normalize_chw01_batch(chw01_batch)

        torch = self._torch
        with torch.cuda.stream(self.stream):
            # 1. Pageable host RAM → page-locked staging buffer.
            self.host_buffer[:n].copy_(torch.from_numpy(np.ascontiguousarray(chw01_batch)))
            # 2. Pinned staging → VRAM via non-blocking DMA (overlaps compute).
            self.device_buffer[:n].copy_(self.host_buffer[:n], non_blocking=True)
            # 3. Fused normalisation on the same dedicated stream.
            normalized = (self.device_buffer[:n] - self._mean) / self._std
        # 4. Single synchronisation point before handing tensors to ONNX.
        self.stream.synchronize()
        return normalized.cpu().numpy()

    def pin_host_tensor(self, tensor):
        """Return a page-locked copy of *tensor* (CUDA builds) or the tensor as-is."""
        if self.cuda_available:
            return tensor.pin_memory()
        return tensor


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
        # Page-locked staging buffer + dedicated CUDA stream for image batches.
        self.image_buffer = PinnedImageBatchBuffer()
        logger.info(
            "inference_engine_ready",
            text_model=text_model_path,
            image_model=image_model_path,
            pinned_dma=self.image_buffer.pinned_memory_used,
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
        (text_logits, text_embeddings, _), (
            image_logits,
            image_embeddings,
            fetch_failures,
        ) = await asyncio.gather(
            self._run_text_inference(texts),
            self._run_image_inference(image_urls),
        )

        # Compute total latency ONCE after both pipelines complete so every
        # ModerationResult in the batch carries the same correct value.
        total_latency_ms = (time.perf_counter() - start) * 1000.0

        results: List[ModerationResult] = []
        for i, item in enumerate(batch):
            text_prob: float = float(sigmoid(text_logits[i : i + 1])[0, 1])
            image_prob: float = float(sigmoid(image_logits[i : i + 1])[0, 1])
            confidence: float = max(text_prob, image_prob)
            label: str = "Toxic" if confidence >= 0.5 else "Non-Toxic"

            results.append(
                ModerationResult(
                    payload_id=item.payload_id,
                    label=label,
                    confidence=confidence,
                    text_confidence=text_prob,
                    image_confidence=image_prob,
                    text_embedding=normalize_l2(text_embeddings[i]).tolist(),
                    image_embedding=normalize_l2(image_embeddings[i]).tolist(),
                    latency_ms=total_latency_ms,
                    image_fetch_failed=fetch_failures[i],
                )
            )

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
    ) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
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

        loop = asyncio.get_running_loop()
        logits, embeddings = await loop.run_in_executor(None, _run)
        # Return a dummy third element so both gather arms unpack uniformly.
        return logits, embeddings, [False] * len(texts)

    async def _run_image_inference(
        self, image_urls: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
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

        # Decode successful fetches to raw CHW[0,1]; track failures separately
        # (failed rows keep the legacy "all-zero = neutral embedding" semantics
        # and are *not* ImageNet-normalised).
        decoded: List[Optional[np.ndarray]] = []
        fetch_failures: List[bool] = []
        for raw in image_bytes_list:
            if raw is None:
                decoded.append(None)
                fetch_failures.append(True)
            else:
                decoded.append(decode_image_to_chw01(raw))
                fetch_failures.append(False)

        pixel_values = np.zeros((len(decoded), 3, 224, 224), dtype=np.float32)
        good_rows = [d for d in decoded if d is not None]
        if good_rows:
            # Pinned host buffer → non-blocking DMA → normalised batch (see
            # PinnedImageBatchBuffer); falls back to NumPy on CPU-only builds.
            normalized_good = self.image_buffer.normalize_batch(np.stack(good_rows, axis=0))
            row_iter = iter(normalized_good)
            for i, d in enumerate(decoded):
                if d is not None:
                    pixel_values[i] = next(row_iter)

        session = self._image_session

        def _run() -> Tuple[np.ndarray, np.ndarray]:
            outputs = session.run(
                ["logits", "embeddings"],
                {"pixel_values": pixel_values},
            )
            return outputs[0], outputs[1]

        loop = asyncio.get_running_loop()
        logits, embeddings = await loop.run_in_executor(None, _run)
        return logits, embeddings, fetch_failures


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
