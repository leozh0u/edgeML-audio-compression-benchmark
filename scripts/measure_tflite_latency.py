"""
Fill the latency axis for the deployable INT8 tflite models.

benchmark.py reports host-CPU single-sample latency as a portable proxy for the
fp32 (torch) and ptq (onnxruntime) variants, but the tflite_int8 rows had none.
This measures the same proxy for the clean int8 tflites via tf.lite.Interpreter
(single thread, batch 1, int8 in/out — the exact graph that ships to the ESP32)
and writes int8_latency_ms back into results/tflite_results.json, which
benchmark.py then merges.

Same caveat as the rest of the harness: this is HOST-CPU latency, comparable
across variants on this machine, not a device number. Real ESP32-S3 latency is
measured on-board and added separately.

Run with the tflite_env python (needs tensorflow):
    ../tflite_env/bin/python3 scripts/measure_tflite_latency.py
"""

import json
import os
import time

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_DIR = os.path.join(REPO_ROOT, "tflite_models")
RESULTS = os.path.join(REPO_ROOT, "results", "tflite_results.json")

WARMUP = 10
RUNS = 200          # tflite invokes are fast; more runs = stabler mean


def latency_ms(path: str) -> float:
    # num_threads=1: the ESP32 is single-core for this workload, and single-thread
    # keeps the proxy comparable to the torch/onnxruntime numbers (batch 1).
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.random.randint(-128, 128, size=inp["shape"], dtype=np.int8)
    interp.set_tensor(inp["index"], x)
    for _ in range(WARMUP):
        interp.invoke()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        interp.invoke()
    return (time.perf_counter() - t0) / RUNS * 1000


def main():
    rows = json.load(open(RESULTS))
    for r in rows:
        path = os.path.join(REPO_ROOT, r["path"])
        ms = latency_ms(path)
        r["int8_latency_ms"] = round(ms, 3)
        print(f"{r['name']:14} {os.path.basename(path):32} {ms:.3f} ms")
    with open(RESULTS, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"updated {RESULTS}")


if __name__ == "__main__":
    main()
