"""StreamShield AI — file-upload moderation engine (video & image).

Executes frame-by-frame inference over uploaded media using PyAV (H.264/MP4
decode; OpenCV used opportunistically when installed) and returns a temporal
risk timeline plus the standardized explainable moderation report.

Shared by:
* ``POST /moderate/upload`` in ``api/webrtc_server.py``
* The Streamlit cockpit's **Upload Moderator** tab
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.moderation_report import build_moderation_report, classify_alert_level

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SUPPORTED_EXTS = SUPPORTED_VIDEO_EXTS | SUPPORTED_IMAGE_EXTS

MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MB
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_MAX_FRAMES = 64
INFER_BATCH_SIZE = 16

# Optional OpenCV acceleration (resize path); PyAV remains the decoder.
try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False


class UploadModerationError(ValueError):
    """Client-facing validation/decoding error carrying an HTTP status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def validate_upload(filename: str, data: bytes) -> str:
    """Validate extension + size; return the normalised lowercase extension."""
    if not filename or "." not in filename:
        raise UploadModerationError(400, "A filename with an extension is required.")
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTS:
        raise UploadModerationError(
            415,
            f"Unsupported file type '{ext}'. Allowed: video {sorted(SUPPORTED_VIDEO_EXTS)}, "
            f"image {sorted(SUPPORTED_IMAGE_EXTS)}.",
        )
    if not data:
        raise UploadModerationError(400, "Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadModerationError(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    return ext


# ---------------------------------------------------------------------------
# Frame extraction (PyAV primary, optional OpenCV pre-resize)
# ---------------------------------------------------------------------------


def frame_to_chw01(frame_hwc: np.ndarray) -> np.ndarray:
    """Resize an ``(H, W, 3)`` uint8 RGB frame to ``(3, 224, 224)`` float32 [0, 1]."""
    if CV2_AVAILABLE:
        resized = cv2.resize(frame_hwc, (224, 224), interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.array(Image.fromarray(frame_hwc).resize((224, 224), Image.BILINEAR))
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)


def decode_video_frames(
    data: bytes,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
    """Decode a MP4/MOV byte stream with PyAV and sample frames at *sample_fps*.

    Returns ``(frames, timestamps, meta)`` — frames are RGB uint8 arrays,
    timestamps are presentation times in seconds, meta carries ``source_fps``,
    ``duration_sec`` and ``frames_decoded``.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - av is a hard dependency
        raise UploadModerationError(503, "PyAV (av) is not installed; video decode unavailable.") from exc

    try:
        container = av.open(io.BytesIO(data))
    except Exception as exc:
        raise UploadModerationError(422, f"Unable to decode video container: {exc}") from exc

    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    frames_decoded = 0
    src_fps = 25.0
    duration = 0.0
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise UploadModerationError(422, "Uploaded video contains no video stream.")
        stream.thread_type = "AUTO"
        src_fps = float(stream.average_rate) if stream.average_rate else 25.0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        stride = max(1, int(round(src_fps / max(sample_fps, 0.1))))

        for idx, frame in enumerate(container.decode(stream)):
            frames_decoded = idx + 1
            if idx % stride != 0:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(round(idx / src_fps, 4))
            if len(frames) >= max_frames:
                break
    finally:
        container.close()

    if not frames:
        raise UploadModerationError(422, "No decodable video frames found in the upload.")
    return frames, timestamps, {
        "source_fps": round(src_fps, 3),
        "duration_sec": round(duration, 3),
        "frames_decoded": frames_decoded,
        "sample_fps": round(sample_fps, 3),
    }


def decode_image_frame(data: bytes) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
    """Decode an uploaded JPEG/PNG into a single-frame list."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise UploadModerationError(422, f"Unable to decode image: {exc}") from exc
    return [np.array(img)], [0.0], {
        "source_fps": 0.0,
        "duration_sec": 0.0,
        "frames_decoded": 1,
        "sample_fps": 0.0,
        "width": img.width,
        "height": img.height,
    }


# ---------------------------------------------------------------------------
# Core async moderation pipeline
# ---------------------------------------------------------------------------

# infer_fn(texts, frames) -> (scores, latency_ms)
# score dict keys: label, confidence, text_confidence, image_confidence
InferFn = Callable[[List[str], List[np.ndarray]], Awaitable[Tuple[List[Dict[str, Any]], float]]]


def _default_frame_text(caption: str, index: int, filename: str) -> str:
    if caption and caption.strip():
        return caption.strip()
    return f"Live media frame {index} from uploaded file {filename}."


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


async def moderate_media_file(
    filename: str,
    data: bytes,
    *,
    infer_fn: InferFn,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    caption: str = "",
    capture_thumbnail: bool = False,
) -> Dict[str, Any]:
    """Moderate an uploaded video/image and build the temporal risk timeline.

    Returns ``{filename, media_type, kind, timeline, report, stats}`` where
    ``report`` is the standardized explainable moderation JSON and ``timeline``
    is a per-frame temporal risk series (see module docstring).
    """
    t_start = time.perf_counter()
    ext = validate_upload(filename, data)
    kind = "video" if ext in SUPPORTED_VIDEO_EXTS else "image"

    # CPU-bound decode runs off the event loop (hardening for the live server).
    if kind == "video":
        frames, timestamps, meta = await asyncio.to_thread(
            decode_video_frames, data, sample_fps=sample_fps, max_frames=max_frames
        )
    else:
        frames, timestamps, meta = await asyncio.to_thread(decode_image_frame, data)

    timeline: List[Dict[str, Any]] = []
    batch_latencies: List[float] = []
    worst_idx = 0

    for start in range(0, len(frames), INFER_BATCH_SIZE):
        chunk = frames[start : start + INFER_BATCH_SIZE]
        texts = [_default_frame_text(caption, start + j, filename) for j in range(len(chunk))]
        scores, latency_ms = await infer_fn(texts, chunk)
        batch_latencies.append(float(latency_ms))
        for j, score in enumerate(scores):
            frame_idx = start + j
            confidence = float(score["confidence"])
            level = classify_alert_level(confidence)
            point = {
                "frame": frame_idx,
                "t_sec": timestamps[frame_idx] if frame_idx < len(timestamps) else round(frame_idx / 10.0, 4),
                "label": score["label"],
                "confidence": round(confidence, 6),
                "text_confidence": round(float(score["text_confidence"]), 6),
                "image_confidence": round(float(score["image_confidence"]), 6),
                "alert_level": level,
                "flagged": score["label"] == "Toxic",
            }
            timeline.append(point)
            if confidence > timeline[worst_idx]["confidence"]:
                worst_idx = frame_idx

    if not timeline:
        raise UploadModerationError(422, "Moderation pipeline produced no results.")

    peak = timeline[worst_idx]
    report = build_moderation_report(
        label=peak["label"],
        confidence=peak["confidence"],
        text_confidence=peak["text_confidence"],
        image_confidence=peak["image_confidence"],
        alert_level=peak["alert_level"],
        frame_ref=f"frame_{peak['frame']}",
        source="file_upload",
    )

    confidences = [p["confidence"] for p in timeline]
    total_ms = (time.perf_counter() - t_start) * 1000.0
    duration = timeline[-1]["t_sec"] - timeline[0]["t_sec"]

    stats = {
        "media_kind": kind,
        "frames_processed": len(timeline),
        "batches": len(batch_latencies),
        "peak_risk": round(max(confidences), 6),
        "mean_risk": round(float(np.mean(confidences)), 6),
        "violations": sum(1 for p in timeline if p["flagged"]),
        "critical_frames": sum(1 for p in timeline if p["alert_level"] == "CRITICAL"),
        "inference_p50_ms": round(_percentile(batch_latencies, 50), 3),
        "inference_p99_ms": round(_percentile(batch_latencies, 99), 3),
        "total_pipeline_ms": round(total_ms, 3),
        "peak_frame": peak["frame"],
        "effective_sample_fps": round(len(timeline) / max(duration, 1e-6), 3) if kind == "video" else 1.0,
        **meta,
    }

    result: Dict[str, Any] = {
        "filename": filename,
        "media_type": f"video/{ext.lstrip('.')}" if kind == "video" else f"image/{ext.lstrip('.')}",
        "kind": kind,
        "timeline": timeline,
        "report": report,
        "stats": stats,
    }

    if capture_thumbnail:
        buf = io.BytesIO()
        Image.fromarray(frames[worst_idx]).save(buf, format="JPEG", quality=85)
        result["peak_frame_jpeg_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")

    return result