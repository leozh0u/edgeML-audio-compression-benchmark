"""
Milestone 5: Quantization-Aware Training (QAT) for the students.

Directly tests the central compression question left open by PTQ: can fine-tuning
with fake-quant recover the INT8 accuracy that post-training quantization loses --
especially for TinyStudent, which got WORSE under ONNX Runtime PTQ.

Approach:
  - FX-graph-mode QAT (torch.ao.quantization.quantize_fx). No manual QuantStub /
    fuse surgery: prepare_qat_fx auto-fuses Conv-BN-ReLU and inserts fake-quant
    observers, so the frozen architectures in common.py are used unchanged.
  - qnnpack backend (the ARM / Apple-Silicon / ESP-class target).
  - Fine-tune with the SAME distillation loss used to train the students, so the
    QAT run is a fair continuation of the KD recipe, not a different objective.
  - Compare FP32 vs QAT-INT8 on accuracy and serialized size; write a JSON row
    per model into results/ for the benchmark harness to pick up.

Runs on CPU: qnnpack quantized ops are CPU-only and fake-quant on MPS is flaky.
Uses the precomputed feature cache, so CPU epochs are fast.
"""

import copy
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

import common as C

EPOCHS = 15
LR = 5e-4
BATCH_SIZE = 32
BACKEND = "qnnpack"
DEV = "cpu"                       # QAT + quantized inference both on CPU here


def serialized_size_kb(state_dict) -> float:
    path = f"/tmp/_qat_size_{os.getpid()}.pt"
    torch.save(state_dict, path)
    kb = os.path.getsize(path) / 1024
    os.remove(path)
    return kb


def run_qat(name, spec, teacher, train_loader, val_loader):
    torch.backends.quantized.engine = BACKEND

    # --- FP32 baseline (load the distilled checkpoint) ---
    fp32 = spec["cls"]().to(DEV)
    fp32.load_state_dict(torch.load(C.CKPT_DIR / spec["ckpt"], map_location=DEV))
    fp32.eval()
    fp32_acc = C.evaluate(fp32, val_loader, DEV)
    fp32_kb = serialized_size_kb(fp32.state_dict())

    # --- Prepare QAT graph from a copy of the trained weights ---
    qconfig_mapping = get_default_qat_qconfig_mapping(BACKEND)
    example = torch.randn(1, *C.INPUT_SHAPE)
    model = copy.deepcopy(fp32).train()
    prepared = prepare_qat_fx(model, qconfig_mapping, example)

    optimizer = torch.optim.AdamW(prepared.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Model selection uses the fake-quant model as an INT8 proxy each epoch (cheap,
    # and deepcopy-safe). We convert to true INT8 exactly once, at the end, from the
    # best epoch's weights -- deepcopy of a QAT module after backward is unsupported.
    best_acc, best_sd = 0.0, None
    for epoch in range(EPOCHS):
        prepared.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEV), y.to(DEV)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = prepared(x)
            loss = C.distill_loss(s_logits, t_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        prepared.eval()
        acc = C.evaluate(prepared, val_loader, DEV)   # fake-quant proxy accuracy
        print(f"  [{name}] epoch {epoch+1}/{EPOCHS} fakequant_val_acc={acc:.3f} "
              f"({time.time()-t0:.1f}s)")
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.detach().cpu().clone() for k, v in prepared.state_dict().items()}

    # Rebuild a fresh QAT graph from the clean FP32 weights, load the best epoch's
    # observed weights, then convert once to a real INT8 model.
    final_prepared = prepare_qat_fx(copy.deepcopy(fp32).train(), qconfig_mapping, example)
    final_prepared.load_state_dict(best_sd)
    final_prepared.eval()
    int8_model = convert_fx(final_prepared)
    int8_acc = C.evaluate(int8_model, val_loader, DEV)   # true INT8 accuracy
    torch.save(int8_model.state_dict(), C.CKPT_DIR / f"{name}_qat_int8.pt")
    best_acc = int8_acc
    int8_kb = serialized_size_kb(int8_model.state_dict())
    result = {
        "name": name,
        "stage": "qat",
        "fp32_acc": round(fp32_acc, 4),
        "fp32_kb": round(fp32_kb, 1),
        "int8_acc": round(best_acc, 4),
        "int8_kb": round(int8_kb, 1),
        "acc_delta": round(best_acc - fp32_acc, 4),
        "size_reduction": round(fp32_kb / int8_kb, 2) if int8_kb else None,
    }
    print(f"--- {name} QAT ---")
    print(f"FP32 : {fp32_acc:.3f} acc, {fp32_kb:.1f} KB")
    print(f"QAT8 : {best_acc:.3f} acc, {int8_kb:.1f} KB "
          f"({result['size_reduction']}x smaller, {result['acc_delta']:+.3f} acc)")
    return result


def main():
    train_ds, val_ds = C.load_cached_folds()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    teacher = C.load_teacher(DEV)   # FP32 teacher on CPU for KD targets

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, spec in C.STUDENTS.items():
        results.append(run_qat(name, spec, teacher, train_loader, val_loader))

    out = C.RESULTS_DIR / "qat_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
