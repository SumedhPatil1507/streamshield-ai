"""Unit and integration tests for api/webrtc_server.py.

Covers REST/WebSocket surfaces, the ``POST /moderate/upload`` file engine,
the platform-standard explainable moderation report contract, and the
**< 20 ms p99 orchestration-overhead SLA** (queue -> broadcast-ready alert
JSON) for batch sizes 1..16 — measured with a deterministic stub engine so the
SLA assertion isolates the platform pipeline from model kernels.
"""

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure offline test execution
os.environ["STREAMSHIELD_OFFLINE"] = "1"

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.webrtc_server import (
    app,
    alert_manager,
    frame_queue,
    stream_manager,
    QueuedFrame,
    ModerationAlertPayload,
    VideoFrameIngestionQueue,
    WebSocketAlertManager,
)
from src.inference_engine import EngineInferenceResult
from src.moderation_report import REPORT_FIELDS, build_moderation_report

SLA_TARGET_P99_MS = 20.0
SLA_BATCH_SIZES = (1, 4, 8, 16)


def _percentile(values, q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


class _StubInferenceEngine:
    """Deterministic in-memory engine stub (no ONNX sessions, no model files).

    Lets the alert-pipeline latency test isolate the *platform* overhead
    (queueing, fusion, alert construction, JSON serialisation readiness)
    from model kernels.
    """

    async def infer_batch(self, texts, images, payload_ids=None):
        results = []
        for i in range(len(texts)):
            confidence = 0.92 if i % 2 == 0 else 0.18
            results.append(
                EngineInferenceResult(
                    payload_id=(payload_ids or [f"stub_{j}" for j in range(len(texts))])[i],
                    label="Toxic" if confidence >= 0.5 else "Non-Toxic",
                    confidence=confidence,
                    text_confidence=confidence,
                    image_confidence=confidence,
                    text_embedding=[0.0] * 128,
                    image_embedding=[0.0] * 128,
                    latency_ms=0.01,
                    provider="StubExecutionProvider",
                )
            )
        return results


def _make_test_mp4(num_frames: int = 12, fps: int = 8):
    """Build a tiny H.264 MP4 in-memory; returns None if no encoder is available."""
    try:
        import av
    except ImportError:
        return None
    buf = io.BytesIO()
    try:
        with av.open(buf, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for i in range(num_frames):
                arr = np.full((64, 64, 3), (i * 18) % 255, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        return None
    data = buf.getvalue()
    return data if len(data) > 100 else None


class TestWebRTCServerREST(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "webrtc_rtsp_ingest")
        self.assertIn("total_frames_inferred", data)

    def test_webrtc_offer_endpoint(self):
        offer_payload = {
            "sdp": "v=0\r\no=- 123456 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n",
            "type": "offer",
            "stream_id": "test_stream_001",
        }
        response = self.client.post("/offer", json=offer_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["stream_id"], "test_stream_001")
        self.assertIn("sdp", data)
        self.assertIn("type", data)

    def test_rtsp_connect_and_disconnect(self):
        connect_payload = {
            "rtsp_url": "rtsp://127.0.0.1:8554/live",
            "stream_id": "rtsp_test_unit",
            "fps_sample_rate": 5.0,
        }
        response = self.client.post("/rtsp/connect", json=connect_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "connected")
        self.assertEqual(data["stream_id"], "rtsp_test_unit")

        # Check streams endpoint
        streams_res = self.client.get("/streams")
        self.assertEqual(streams_res.status_code, 200)
        streams = streams_res.json()
        self.assertTrue(any(s["stream_id"] == "rtsp_test_unit" for s in streams))

        # Disconnect
        disc_res = self.client.post("/rtsp/disconnect/rtsp_test_unit")
        self.assertEqual(disc_res.status_code, 200)
        self.assertEqual(disc_res.json()["status"], "disconnected")


class TestVideoFrameQueueAndAlerts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_alert_manager = WebSocketAlertManager()
        self.test_queue = VideoFrameIngestionQueue(
            alert_manager=self.test_alert_manager,
            max_queue_size=16,
            batch_size=4,
            batch_timeout_ms=10.0,
        )
        await self.test_queue.start()

    async def asyncTearDown(self):
        await self.test_queue.stop()

    async def test_enqueue_and_batch_processing(self):
        # Push 4 frames
        for i in range(4):
            frame = QueuedFrame(
                stream_id="test_stream",
                frame_id=i + 1,
                timestamp=100.0 + i,
                image_data=np.zeros((224, 224, 3), dtype=np.uint8),
                associated_text=f"Test message frame {i+1}",
            )
            success = self.test_queue.enqueue_frame_nowait(frame)
            self.assertTrue(success)

        # Wait for worker to finish processing the batch
        for _ in range(30):
            if self.test_queue.total_frames_inferred >= 4:
                break
            await asyncio.sleep(0.1)

        self.assertGreaterEqual(self.test_queue.total_frames_inferred, 4)

    def test_websocket_alerts_subscription(self):
        client = TestClient(app)
        with client.websocket_connect("/ws/alerts") as websocket:
            websocket.send_text(json.dumps({"action": "ping"}))
            response = websocket.receive_text()
            data = json.loads(response)
            self.assertEqual(data["action"], "pong")


class TestStandardizedModerationReport(unittest.TestCase):
    """The explainable JSON contract shared by alerts, uploads and the cockpit."""

    def test_report_schema_exact_fields(self):
        report = build_moderation_report(
            "Toxic", 0.942, 0.918, 0.942,
            source="webrtc", stream_id="w1", frame_ref="1420",
        )
        self.assertEqual(tuple(report.keys()), REPORT_FIELDS)
        self.assertIn("CRITICAL", report["verdict"])
        self.assertIn("Vision: 94.2%", report["reasoning"])
        self.assertIn("Text/Audio: 91.8%", report["reasoning"])
        self.assertEqual(len(report["next_steps"]), 3)
        self.assertIn("PostgreSQL audit trail", report["next_steps"][-1])

    def test_alert_payload_carries_report_fields(self):
        alert = ModerationAlertPayload(
            stream_id="s", frame_id=1, label="Toxic", confidence=0.9,
            text_confidence=0.88, image_confidence=0.9, latency_ms=12.0,
            alert_level="CRITICAL", flagged=True,
            verdict="V", reasoning="R", recommendation="Rec", next_steps=["a", "b", "c"],
        )
        d = alert.model_dump()
        for field in REPORT_FIELDS:
            self.assertIn(field, d)


class TestAlertPipelineLatencySLA(unittest.IsolatedAsyncioTestCase):
    """< 20 ms p99 orchestration overhead (queue -> broadcast-ready alert)."""

    async def asyncSetUp(self):
        self.queue = VideoFrameIngestionQueue(
            alert_manager=WebSocketAlertManager(),
            inference_engine=_StubInferenceEngine(),
            max_queue_size=64,
            batch_size=16,
        )

    async def test_p99_alert_pipeline_under_20ms_batch_up_to_16(self):
        for bs in SLA_BATCH_SIZES:
            batch = [
                QueuedFrame(
                    stream_id="sla_stream", frame_id=i, timestamp=1000.0 + i,
                    image_data=np.zeros((224, 224, 3), dtype=np.uint8),
                    associated_text=f"sla pipeline message {i}",
                )
                for i in range(bs)
            ]
            for _ in range(3):  # warm caches / task pools
                await self.queue._process_batch(batch)
            latencies = []
            for _ in range(12):
                t0 = time.perf_counter()
                await self.queue._process_batch(batch)
                latencies.append((time.perf_counter() - t0) * 1000.0)
            p99 = _percentile(latencies, 99)
            self.assertLess(
                p99, SLA_TARGET_P99_MS,
                f"Alert pipeline orchestration p99={p99:.2f}ms exceeds 20ms at batch {bs}",
            )


class TestModerateUploadEndpoint(unittest.TestCase):
    """POST /moderate/upload — file engine + temporal risk timeline + report."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_upload_png_image(self):
        buf = io.BytesIO()
        Image.new("RGB", (120, 90), (10, 120, 60)).save(buf, format="PNG")
        res = self.client.post(
            "/moderate/upload",
            files={"file": ("frame.png", buf.getvalue(), "image/png")},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["kind"], "image")
        self.assertEqual(tuple(data["report"].keys()), REPORT_FIELDS)
        self.assertEqual(len(data["timeline"]), 1)
        self.assertIn("peak_frame_jpeg_b64", data)
        self.assertIn("inference_p99_ms", data["stats"])
        self.assertIn("frames_processed", data["stats"])

    def test_upload_rejects_unsupported_extension(self):
        res = self.client.post(
            "/moderate/upload",
            files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        self.assertEqual(res.status_code, 415)

    def test_upload_rejects_undecodable_video(self):
        res = self.client.post(
            "/moderate/upload",
            files={"file": ("broken.mp4", b"this is not a real mp4 container", "video/mp4")},
        )
        self.assertEqual(res.status_code, 422)

    def test_upload_mp4_video_temporal_timeline(self):
        mp4_bytes = _make_test_mp4(num_frames=12, fps=8)
        if not mp4_bytes:
            self.skipTest("libx264 encoder unavailable in this PyAV build")
        res = self.client.post(
            "/moderate/upload",
            files={"file": ("clip.mp4", mp4_bytes, "video/mp4")},
            data={"sample_fps": "6.0", "max_frames": "16", "caption": "toxic hateful stream test"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["kind"], "video")
        self.assertGreaterEqual(len(data["timeline"]), 2)
        self.assertEqual(tuple(data["report"].keys()), REPORT_FIELDS)
        timestamps = [p["t_sec"] for p in data["timeline"]]
        self.assertEqual(timestamps, sorted(timestamps))
        for point in data["timeline"]:
            self.assertIn(point["alert_level"], ("SAFE", "WARNING", "CRITICAL"))
            self.assertTrue(0.0 <= point["confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
