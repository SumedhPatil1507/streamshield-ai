"""Instrumented micro-benchmark of the alert pipeline stages."""
import asyncio
import gc
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
os.environ["STREAMSHIELD_OFFLINE"] = "1"

import numpy as np  # noqa: E402

from api.webrtc_server import (  # noqa: E402
    QueuedFrame,
    VideoFrameIngestionQueue,
    WebSocketAlertManager,
)
from tests.test_webrtc_server import _StubInferenceEngine  # noqa: E402


async def main():
    queue = VideoFrameIngestionQueue(
        alert_manager=WebSocketAlertManager(),
        inference_engine=_StubInferenceEngine(),
        max_queue_size=64,
        batch_size=16,
    )
    for bs in (1, 4, 8, 16):
        batch = [
            QueuedFrame(
                stream_id="s", frame_id=i, timestamp=1000.0 + i,
                image_data=np.zeros((224, 224, 3), dtype=np.uint8),
                associated_text=f"msg {i}",
            )
            for i in range(bs)
        ]
        for _ in range(3):
            await queue._process_batch(batch)

        lat = []
        for _ in range(12):
            gc.collect()
            t0 = time.perf_counter()
            await queue._process_batch(batch)
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat_gc_on = sorted(lat)

        gc.disable()
        lat = []
        for _ in range(12):
            t0 = time.perf_counter()
            await queue._process_batch(batch)
            lat.append((time.perf_counter() - t0) * 1000.0)
        gc.enable()
        lat_gc_off = sorted(lat)

        print(
            f"bs={bs:2d} gc_on  min={lat_gc_on[0]:7.3f} p50={lat_gc_on[6]:7.3f} p99={lat_gc_on[-1]:7.3f} | "
            f"gc_off min={lat_gc_off[0]:7.3f} p50={lat_gc_off[6]:7.3f} p99={lat_gc_off[-1]:7.3f} ms",
            flush=True,
        )


asyncio.run(main())
