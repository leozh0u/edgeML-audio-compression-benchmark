"""
Milestone 7: Benchmark harness -> the differentiator of this project.

Sweeps every model variant produced by the pipeline and emits ONE unified table
plus the accuracy-vs-size and accuracy-vs-latency Pareto frontiers.

Sources it merges:
  - FP32 torch models (teacher, mid, tiny): evaluated + measured live here.
  - PTQ INT8 (ONNX Runtime):   results/ptq_results.json   (quantize_onnxruntime.py)
  - QAT INT8:                  results/qat_results.json    (qat_student.py)
  - Pruned:                    results/prune_results.json  (prune_student.py)

Latency is HOST-CPU latency (single-sample, batch 1), a portable proxy. Real
on-device ESP32-S3 latency is a separate, later milestone; those rows get added
to benchmark.json once the board is flashed.

Outputs:
  results/benchmark.json   full merged table
  results/pareto.png       accuracy vs size, with Pareto frontier highlighted
"""

import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import common as C

DEV_EVAL = C.device()      # accuracy eval can use MPS
LAT_RUNS = 50


def host_latency_ms(model, dev="cpu", runs=LAT_RUNS):
    """Mean single-sample forward latency in ms on the given device."""
    model = model.to(dev).eval()
    x = torch.randn(1, *C.INPUT_SHAPE, device=dev)
    with torch.no_grad():
        for _ in range(5):           # warmup
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if dev == "mps":
            torch.mps.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def serialized_kb(model):
    import os
    p = f"/tmp/_bench_{os.getpid()}.pt"
    torch.save(model.state_dict(), p)
    kb = os.path.getsize(p) / 1024
    os.remove(p)
    return kb


def fp32_rows(val_loader):
    """Evaluate the three FP32 torch models live."""
    rows = []
    variants = [
        ("teacher", C.CNNTeacher, "teacher_v2_best.pt"),
        ("mid_student", C.MidStudent, "student_mid_best.pt"),
        ("tiny_student", C.TinyStudent, "student_distilled_best.pt"),
    ]
    for name, cls, ckpt in variants:
        m = cls()
        m.load_state_dict(torch.load(C.CKPT_DIR / ckpt, map_location="cpu"))
        acc = C.evaluate(m.to(DEV_EVAL), val_loader, DEV_EVAL)
        rows.append({
            "name": name, "stage": "fp32",
            "params": C.count_params(m),
            "acc": round(acc, 4),
            "size_kb": round(serialized_kb(m), 1),
            "latency_ms": round(host_latency_ms(m, "cpu"), 3),
        })
        print(f"  fp32 {name:12} acc={acc:.3f} "
              f"size={rows[-1]['size_kb']}KB lat={rows[-1]['latency_ms']}ms")
    return rows


def load_json(path):
    return json.load(open(path)) if path.exists() else []


def pareto_front(points, x_key, y_key, minimize_x=True, maximize_y=True):
    """Return the subset of points on the Pareto frontier (small x, large y)."""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            better_x = (q[x_key] <= p[x_key]) if minimize_x else (q[x_key] >= p[x_key])
            better_y = (q[y_key] >= p[y_key]) if maximize_y else (q[y_key] <= p[y_key])
            strict = (q[x_key] != p[x_key]) or (q[y_key] != p[y_key])
            if better_x and better_y and strict:
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[x_key])


def normalize(rows):
    """Map heterogeneous per-stage rows to a common schema (name, stage, acc, size_kb)."""
    out = []
    for r in rows:
        stage = r.get("stage")
        if stage in ("fp32",):
            out.append({"label": f"{r['name']}-fp32", **r})
        elif stage == "ptq":
            out.append({"label": f"{r['name']}-ptq", "name": r["name"], "stage": "ptq",
                        "acc": r["int8_acc"], "size_kb": r["int8_kb"],
                        "latency_ms": r.get("int8_latency_ms")})
        elif stage == "qat":
            out.append({"label": f"{r['name']}-qat", "name": r["name"], "stage": "qat",
                        "acc": r["int8_acc"], "size_kb": r["int8_kb"],
                        "latency_ms": r.get("int8_latency_ms")})
        elif stage == "tflite_int8":
            out.append({"label": f"{r['name']}-tflite", "name": r["name"], "stage": "tflite_int8",
                        "acc": r["int8_acc"], "size_kb": r["int8_kb"],
                        "latency_ms": r.get("int8_latency_ms")})
        elif stage == "prune":
            out.append({"label": f"{r['name']}-prune{int(r['target_sparsity']*100)}",
                        "name": r["name"], "stage": "prune",
                        "acc": r["acc"], "size_kb": r["compressed_kb"],
                        "latency_ms": None})
    return out


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"fp32": "#1f77b4", "ptq": "#ff7f0e", "qat": "#2ca02c", "prune": "#d62728",
              "tflite_int8": "#9467bd"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for stage, c in colors.items():
        pts = [r for r in rows if r["stage"] == stage]
        if pts:
            ax.scatter([p["size_kb"] for p in pts], [p["acc"] * 100 for p in pts],
                       c=c, label=stage.upper(), s=60, zorder=3)

    front = pareto_front(rows, "size_kb", "acc")
    ax.plot([p["size_kb"] for p in front], [p["acc"] * 100 for p in front],
            "k--", alpha=0.6, zorder=2, label="Pareto frontier")
    for p in rows:
        ax.annotate(p["label"], (p["size_kb"], p["acc"] * 100),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Serialized size (KB, log scale)")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("ESC-50 edge models: accuracy vs size (fold-5 val)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, val_ds = C.load_cached_folds()
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    rows = []
    rows += fp32_rows(val_loader)
    rows += load_json(C.RESULTS_DIR / "ptq_results.json")
    rows += load_json(C.RESULTS_DIR / "qat_results.json")
    rows += load_json(C.RESULTS_DIR / "prune_results.json")
    rows += load_json(C.RESULTS_DIR / "tflite_results.json")

    unified = normalize(rows)
    with open(C.RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(unified, f, indent=2)

    print("\n=== Unified benchmark table ===")
    print(f"{'model':28} {'acc%':>6} {'size_kb':>9} {'lat_ms':>8}")
    for r in sorted(unified, key=lambda r: r["size_kb"]):
        lat = f"{r['latency_ms']:.2f}" if r.get("latency_ms") else "  -"
        print(f"{r['label']:28} {r['acc']*100:6.1f} {r['size_kb']:9.1f} {lat:>8}")

    front = pareto_front(unified, "size_kb", "acc")
    print("\n=== Pareto frontier (accuracy vs size) ===")
    for r in front:
        print(f"  {r['label']:28} {r['acc']*100:5.1f}%  {r['size_kb']:.1f} KB")

    plot(unified, C.RESULTS_DIR / "pareto.png")


if __name__ == "__main__":
    main()
