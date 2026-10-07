"""Unit tests for UltraLowLatencyInferenceEngine and PinnedFrameBatchBuffer."""

import asyncio
import os
import unittest
import numpy as np
import torch
from PIL import Image

# Ensure offline mode for test suite
os.environ["STREAMSHIELD_OFFLINE"] = "1"

from src.inference_engine import (
    PinnedFrameBatchBuffer,
    UltraLowLatencyInferenceEngine,
    EngineInferenceResult,
    get_inference_engine,
)
from src.streaming.schemas import MultimodalPayload


class TestPinnedFrameBatchBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = PinnedFrameBatchBuffer(max_batch_size=8)

    def test_buffer_allocation(self):
        self.assertIsNotNone(self.buffer.pinned_cpu_buffer)
        self.assertEqual(self.buffer.pinned_cpu_buffer.shape, (8, 3, 224, 224))

    def test_async_load_and_transfer_frames(self):
        # Create 3 synthetic test images
        img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img2 = np.zeros((224, 224, 3), dtype=np.uint8)
        img3 = Image.new("RGB", (100, 100), color=(0, 255, 0))  # resized

        tensor_frames, failures, transfer_ms = self.buffer.async_load_and_transfer_frames([img1, img2, img3])
        self.assertEqual(tensor_frames.shape, (3, 3, 224, 224))
        self.assertEqual(failures, [False, False, False])
        self.assertGreaterEqual(transfer_ms, 0.0)


class TestUltraLowLatencyInferenceEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = get_inference_engine()

    async def test_infer_batch_multimodal(self):
        texts = [
            "This is a clean and friendly message.",
            "I hate everything and this stream is toxic garbage.",
        ]
        images = [
            Image.new("RGB", (224, 224), color=(0, 128, 0)),
            Image.new("RGB", (224, 224), color=(128, 0, 0)),
        ]

        results = await self.engine.infer_batch(texts, images)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, EngineInferenceResult)
            self.assertIn(r.label, ["Toxic", "Non-Toxic"])
            self.assertTrue(0.0 <= r.confidence <= 1.0)
            self.assertEqual(len(r.text_embedding), 128)
            self.assertEqual(len(r.image_embedding), 128)
            self.assertGreater(r.latency_ms, 0.0)

    async def test_run_batch_schema_compatibility(self):
        payloads = [
            MultimodalPayload(text_content="Clean chat payload", image_url="https://streamshield.ai/test.jpg"),
        ]
        results = await self.engine.run_batch(payloads)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].payload_id, payloads[0].payload_id)


if __name__ == "__main__":
    unittest.main()
