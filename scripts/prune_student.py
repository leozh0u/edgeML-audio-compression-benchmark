"""
Milestone 6: Magnitude pruning + fine-tune for the students.

Adds pruning points to the Pareto frontier. Uses global unstructured L1 pruning
across all conv/linear weights, then fine-tunes with the distillation loss to
recover accuracy. We sweep several sparsity levels so the writeup can show the
accuracy-vs-sparsity curve (where each model's accuracy falls off a cliff).

Note on size: unstructured pruning zeros weights but does NOT shrink the dense
tensor -- on-disk size only drops if stored sparse/compressed. We therefore report
sparsity and the compressed (zlib) size, which is what actually matters for flash
footprint, alongside accuracy. Structured/channel pruning (real dense speedup) is
noted as future work in DECISIONS.md.
"""

import copy
import json
import os
import time
import zlib

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

import common as C

SPARSITIES = [0.3, 0.5, 0.7, 0.9]
FINETUNE_EPOCHS = 10
LR = 5e-4
BATCH_SIZE = 32
DEV = C.device()


def prunable_params(model):
    return [(m, "weight") for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def measure_sparsity(model):
    zeros = total = 0
    for m, _ in prunable_params(model):
        w = m.weight
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros / total


def compressed_size_kb(model):
    """zlib-compressed state_dict size -- proxy for flash footprint of a sparse model."""
    path = f"/tmp/_prune_{os.getpid()}.pt"
    torch.save(model.state_dict(), path)
    raw = open(path, "rb").read()
    os.remove(path)
    return len(zlib.compress(raw, 9)) / 1024


def finetune(model, teacher, train_loader, val_loader):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)
    best_acc, best_sd = 0.0, None
    for _ in range(FINETUNE_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEV), y.to(DEV)
            with torch.no_grad():
                t_logits = teacher(x)
            loss = C.distill_loss(model(x), t_logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        acc = C.evaluate(model, val_loader, DEV)
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_acc, best_sd


def prune_at(name, spec, sparsity, teacher, train_loader, val_loader):
    model = spec["cls"]().to(DEV)
    model.load_state_dict(torch.load(C.CKPT_DIR / spec["ckpt"], map_location=DEV))

    params = prunable_params(model)
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)

    t0 = time.time()
    acc, best_sd = finetune(model, teacher, train_loader, val_loader)

    # Bake the mask into the weights so the zeros are permanent, then measure.
    for m, n in params:
        prune.remove(m, n)
    real_sparsity = measure_sparsity(model)
    size_kb = compressed_size_kb(model)

    print(f"  [{name}] sparsity~{sparsity:.0%} (actual {real_sparsity:.1%}) "
          f"acc={acc:.3f} zlib={size_kb:.1f}KB ({time.time()-t0:.0f}s)")
    return {
        "name": name, "stage": "prune",
        "target_sparsity": sparsity,
        "actual_sparsity": round(real_sparsity, 4),
        "acc": round(acc, 4),
        "compressed_kb": round(size_kb, 1),
    }


def main():
    train_ds, val_ds = C.load_cached_folds()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    teacher = C.load_teacher(DEV)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, spec in C.STUDENTS.items():
        for s in SPARSITIES:
            results.append(prune_at(name, spec, s, teacher, train_loader, val_loader))

    out = C.RESULTS_DIR / "prune_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
