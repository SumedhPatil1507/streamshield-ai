"""Profile ONNX Runtime thread settings for the text moderation model."""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

MODEL = root / "models" / "text_toxicity.onnx"
VISION = root / "models" / "image_classifier.onnx"


def bench(session, feed, n=10, warmup=3):
    for _ in range(warmup):
        session.run(None, feed)
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, feed)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return lat[n // 2], lat[0], lat[-1]


def main():
    print(f"cpu logical cores: {__import__('os').cpu_count()}")
    ids = np.random.randint(0, 30522, (16, 128)).astype(np.int64)
    mask = np.ones((16, 128), dtype=np.int64)
    pix = np.random.rand(16, 3, 224, 224).astype(np.float32)

    for nth in (1, 2, 4, 6):
        so = ort.SessionOptions()
        so.intra_op_num_threads = nth
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        t_sess = ort.InferenceSession(str(MODEL), sess_options=so, providers=["CPUExecutionProvider"])
        v_sess = ort.InferenceSession(str(VISION), sess_options=so, providers=["CPUExecutionProvider"])
        tm, tmin, tmax = bench(t_sess, {"input_ids": ids, "attention_mask": mask})
        vm, vmin, vmax = bench(v_sess, {"pixel_values": pix})
        print(
            f"threads={nth}  TEXT bs16 p50={tm:7.2f} min={tmin:7.2f} max={tmax:7.2f} ms | "
            f"VISION bs16 p50={vm:7.2f} min={vmin:7.2f} max={vmax:7.2f} ms | "
            f"parallel-ish sum p50={tm + vm:7.2f} ms"
        )


if __name__ == "__main__":
    main()
