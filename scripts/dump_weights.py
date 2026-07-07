"""
Phase 1 of the clean TFLite conversion (runs in esc50env, which has torch).

Dumps each student's trained weights -- plus a few reference FP32 logits -- into
a plain .npz of numpy arrays, so the TF-side converter (convert_tflite_clean.py,
runs in tflite_env) needs only numpy + tensorflow, never torch. This avoids
installing torch into the 3.11 env and keeps the two toolchains decoupled.

For each conv-bn block we store conv weight/bias and BN gamma/beta/mean/var;
plus the final linear. We also store K validation inputs and the torch logits on
them, which convert_tflite_clean.py uses to prove the Keras port is numerically
faithful before it quantizes anything.
"""

import numpy as np
import torch
import torch.nn as nn

import common as C

K_REF = 64   # number of val samples for the parity check


def dump_student(name, spec):
    model = spec["cls"]()
    model.load_state_dict(torch.load(C.CKPT_DIR / spec["ckpt"], map_location="cpu"))
    model.eval()

    arrays = {}
    conv_i = bn_i = 0
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            arrays[f"conv{conv_i}_w"] = m.weight.detach().numpy()   # [O,I,kh,kw]
            arrays[f"conv{conv_i}_b"] = m.bias.detach().numpy()     # [O]
            conv_i += 1
        elif isinstance(m, nn.BatchNorm2d):
            arrays[f"bn{bn_i}_gamma"] = m.weight.detach().numpy()
            arrays[f"bn{bn_i}_beta"] = m.bias.detach().numpy()
            arrays[f"bn{bn_i}_mean"] = m.running_mean.detach().numpy()
            arrays[f"bn{bn_i}_var"] = m.running_var.detach().numpy()
            arrays[f"bn{bn_i}_eps"] = np.array(m.eps, dtype=np.float32)
            bn_i += 1
    arrays["fc_w"] = model.classifier.weight.detach().numpy()       # [O,I]
    arrays["fc_b"] = model.classifier.bias.detach().numpy()
    arrays["n_conv"] = np.array(conv_i)

    # Reference inputs + torch logits for the parity check.
    data = np.load(C.CACHE_PATH)
    feats, labels, folds = data["feats"], data["labels"], data["folds"]
    val_feats = feats[folds == 5][:K_REF]                          # [K,64,216]
    with torch.no_grad():
        x = torch.tensor(val_feats, dtype=torch.float32).unsqueeze(1)  # NCHW
        ref_logits = model(x).numpy()
    arrays["ref_x"] = val_feats
    arrays["ref_logits"] = ref_logits

    out = C.TFLITE_DIR / f"keras_port_{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(f"{name}: dumped {conv_i} conv blocks + fc -> {out.name}  "
          f"(ref {val_feats.shape[0]} samples)")


def main():
    for name, spec in C.STUDENTS.items():
        dump_student(name, spec)


if __name__ == "__main__":
    main()
