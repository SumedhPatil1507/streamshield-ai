"""Unit tests for StreamShield AI multimodal inference engine and model utilities."""

import os
import unittest
import numpy as np

# Force offline mode for deterministic testing without external network calls
os.environ["STREAMSHIELD_OFFLINE"] = "1"

from src.models.inference import InferenceEngine, ModerationResult, tokenize_batch, preprocess_images_batch
from src.models.model_utils import normalize_l2, sigmoid
from src.streaming.config import settings
from src.streaming.schemas import MultimodalPayload


class TestModelUtils(unittest.TestCase):
    def test_sigmoid(self):
        logits = np.array([0.0, 2.0, -2.0], dtype=np.float32)
        probs = sigmoid(logits)
        self.assertAlmostEqual(float(probs[0]), 0.5, places=4)
        self.assertGreater(float(probs[1]), 0.8)
        self.assertLess(float(probs[2]), 0.2)

    def test_normalize_l2(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        norm = normalize_l2(vec)
        self.assertAlmostEqual(float(np.linalg.norm(norm)), 1.0, places=4)
        self.assertAlmostEqual(float(norm[0]), 0.6, places=4)
        self.assertAlmostEqual(float(norm[1]), 0.8, places=4)


class TestInferenceEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = InferenceEngine(settings.TEXT_ONNX_PATH, settings.IMAGE_MODEL_PATH)

    async def test_run_batch_uniform_latency_and_labels(self):
        payloads = [
            MultimodalPayload(text_content="Clean message #1", image_url="https://picsum.photos/224"),
            MultimodalPayload(text_content="Clean message #2", image_url="https://picsum.photos/224"),
            MultimodalPayload(text_content="Clean message #3", image_url="https://picsum.photos/224"),
        ]
        results = await self.engine.run_batch(payloads)
        self.assertEqual(len(results), 3)

        # Ensure all items in the batch record the same batch wall-clock latency
        batch_latency = results[0].latency_ms
        self.assertGreater(batch_latency, 0.0)
        for r in results:
            self.assertEqual(r.latency_ms, batch_latency)
            self.assertIn(r.label, ["Toxic", "Non-Toxic"])
            self.assertTrue(0.0 <= r.confidence <= 1.0)
            self.assertEqual(len(r.text_embedding), 128)
            self.assertEqual(len(r.image_embedding), 128)

    async def test_tokenize_batch(self):
        texts = ["Hello world", "StreamShield content filter"]
        tokens = tokenize_batch(texts)
        self.assertIn("input_ids", tokens)
        self.assertIn("attention_mask", tokens)
        self.assertEqual(tokens["input_ids"].shape, (2, 128))
        self.assertEqual(tokens["attention_mask"].shape, (2, 128))


if __name__ == "__main__":
    unittest.main()
