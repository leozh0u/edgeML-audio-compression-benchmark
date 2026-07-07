"""
Phase 2 of the clean TFLite conversion (runs in tflite_env: TF + numpy only).

Rebuilds each student as a Keras model, ports the trained weights from the .npz
that dump_weights.py produced (esc50env/torch side), and converts to a fully-INT8
.tflite with TFLiteConverter.from_keras_model + a representative dataset.

Why this finally works where milestone 4 hung: every prior attempt went through a
TF *SavedModel-restore-from-disk* step (onnx2tf -osd, from_saved_model) which froze
on this machine. from_keras_model traces an in-memory concrete function instead --
no SavedModel bundle is written or restored, so the hang is structurally avoided.

Guardrails:
  1. FP32 parity: Keras logits must match the torch reference logits (from the npz)
     to a tight tolerance BEFORE we quantize. A wrong weight port is caught here,
     not silently shipped to the device.
  2. INT8 validation: the quantized .tflite is run through tf.lite.Interpreter on
     the full fold-5 val set; we report real accuracy + file size.

Layout note: torch is NCHW, TFLite/Keras is NHWC. Input becomes [64,216,1]. This is
also the natural layout for the ESP32 firmware (mel_buf is [mel][frame], C=1 last),
so no on-device transpose is needed.
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_DIR = os.path.join(REPO_ROOT, "tflite_models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CACHE_PATH = os.path.join(REPO_ROOT, "feature_cache.npz")

N_MELS, N_FRAMES, N_CLASSES = 64, 216, 50
STUDENTS = ["mid_student", "tiny_student"]


def build_keras(npz):
    """Reconstruct the student as a Keras model and load the ported weights."""
    n_conv = int(npz["n_conv"])
    inp = keras.Input(shape=(N_MELS, N_FRAMES, 1))
    x = inp
    for i in range(n_conv):
        out_ch = npz[f"conv{i}_w"].shape[0]
        # torch pad=1, 3x3, stride1  ==  Keras 'same'
        conv = layers.Conv2D(out_ch, 3, padding="same", use_bias=True)
        x = conv(x)
        # torch BN eps default 1e-5; Keras default is 1e-3 -> MUST override
        bn = layers.BatchNormalization(epsilon=float(npz[f"bn{i}_eps"]))
        x = bn(x)
        x = layers.ReLU()(x)
        # Pool after every block except the last (last -> global avg), matching torch
        if i < n_conv - 1:
            x = layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)

        # port weights immediately
        # torch conv [O,I,kh,kw] -> keras [kh,kw,I,O]
        conv.set_weights([
            np.transpose(npz[f"conv{i}_w"], (2, 3, 1, 0)),
            npz[f"conv{i}_b"],
        ])
        bn.set_weights([
            npz[f"bn{i}_gamma"], npz[f"bn{i}_beta"],
            npz[f"bn{i}_mean"], npz[f"bn{i}_var"],
        ])

    x = layers.GlobalAveragePooling2D()(x)           # == AdaptiveAvgPool2d(1)+flatten
    dense = layers.Dense(N_CLASSES)
    out = dense(x)
    dense.set_weights([npz["fc_w"].T, npz["fc_b"]])  # torch [O,I] -> keras [I,O]

    return keras.Model(inp, out)


def check_parity(model, npz):
    ref_x = npz["ref_x"]                              # [K,64,216]
    ref_logits = npz["ref_logits"]                    # [K,50]
    x = ref_x[..., np.newaxis].astype(np.float32)     # NHWC
    keras_logits = model.predict(x, verbose=0)
    max_abs = np.max(np.abs(keras_logits - ref_logits))
    agree = np.mean(keras_logits.argmax(1) == ref_logits.argmax(1))
    print(f"  parity: max|Δlogit|={max_abs:.2e}  argmax agreement={agree:.3f}")
    return max_abs, agree


def representative_dataset(train_feats):
    def gen():
        for i in range(min(200, len(train_feats))):
            x = train_feats[i][np.newaxis, :, :, np.newaxis].astype(np.float32)
            yield [x]
    return gen


def to_int8_tflite(model, train_feats):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(train_feats)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8      # full int8 in/out for TFLite Micro
    conv.inference_output_type = tf.int8
    return conv.convert()


def eval_tflite(model_bytes, val_feats, val_labels):
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zp = inp["quantization"]
    correct = 0
    for feat, label in zip(val_feats, val_labels):
        x = feat[np.newaxis, :, :, np.newaxis].astype(np.float32)
        xq = np.round(x / scale + zp).astype(inp["dtype"])
        interp.set_tensor(inp["index"], xq)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out["index"])[0]))
        correct += int(pred == label)
    return correct / len(val_labels)


def main():
    data = np.load(CACHE_PATH)
    feats, labels, folds = data["feats"], data["labels"], data["folds"]
    train_feats = feats[folds != 5]
    val_feats, val_labels = feats[folds == 5], labels[folds == 5]

    results = []
    for name in STUDENTS:
        print(f"\n=== {name} ===")
        npz = np.load(os.path.join(TFLITE_DIR, f"keras_port_{name}.npz"))
        model = build_keras(npz)
        max_abs, agree = check_parity(model, npz)
        if agree < 0.99 or max_abs > 1e-2:
            print(f"  !! parity FAILED for {name}; not converting. Fix the port first.")
            continue

        tfl = to_int8_tflite(model, train_feats)
        out_path = os.path.join(TFLITE_DIR, f"{name}_int8_clean.tflite")
        with open(out_path, "wb") as f:
            f.write(tfl)
        size_kb = len(tfl) / 1024
        acc = eval_tflite(tfl, val_feats, val_labels)
        print(f"  INT8 tflite: {acc:.3f} acc, {size_kb:.1f} KB -> {os.path.basename(out_path)}")
        results.append({
            "name": name, "stage": "tflite_int8",
            "int8_acc": round(acc, 4), "int8_kb": round(size_kb, 1),
            "parity_max_abs": float(max_abs), "parity_agreement": float(agree),
            "path": os.path.relpath(out_path, REPO_ROOT),
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tflite_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(RESULTS_DIR, 'tflite_results.json')}")


if __name__ == "__main__":
    main()
