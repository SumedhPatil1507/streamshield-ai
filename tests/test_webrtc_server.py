"""Unit and integration tests for api/webrtc_server.py."""

import asyncio
import json
import os
import sys
from pathlib import Path
import unittest
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
