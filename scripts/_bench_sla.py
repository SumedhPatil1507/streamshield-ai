"""Benchmark tuned engine p50/p99 across batch sizes (post thread-fix)."""
import asyncio
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
os.environ.setdefault("STREAMSHIELD_OFFLINE", "1")

from PIL import Image  # noqa: E402

from src.inference_engine import get_inference_engine  # noqa: E402

engine = get_inference_engine()


def percentile(vals, q):
    ordered = sorted(vals)
    idx = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[idx]


async def main():
    texts = ["clean friendly stream message test"] * 16
    imgs = [Image.new("RGB", (224, 224), (30, 60, 90))] * 16
    # warmup
    for _ in range(3):
        await engine.infer_batch(texts[:8], imgs[:8])
    print("batch |  p50_ms |  p99_ms |  min_ms")
    for bs in (1, 4, 8, 16):
        lat = []
        for _ in range(15):
            t0 = time.perf_counter()
            await engine.infer_batch(texts[:bs], imgs[:bs])
            lat.append((time.perf_counter() - t0) * 1000)
        print(f"  {bs:2d}  | {percentile(lat, 50):7.2f} | {percentile(lat, 99):7.2f} | {min(lat):7.2f}")


asyncio.run(main())
