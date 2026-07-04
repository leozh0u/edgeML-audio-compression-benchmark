"""
Milestone 4a: PyTorch -> ONNX -> TensorFlow -> TFLite INT8.
This is where real conv quantization happens (unlike the PyTorch dynamic-quant PTQ pass,
which only touched Linear layers and gave near-noise results).
Run on Mac, no board required. Gives real size + accuracy numbers for the Pareto plot.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import tensorflow as tf

ESC50_ROOT = os.path.expanduser("~/ESC-50")
AUDIO_DIR = os.path.join(ESC50_ROOT, "audio")
META_CSV = os.path.join(ESC50_ROOT, "meta/esc50.csv")
CKPT_DIR = os.path.expanduser("~/edge-ml-esc50/checkpoints")
TFLITE_DIR = os.path.expanduser("~/edge-ml-esc50/tflite_models")
SR = 22050
N_MELS = 64
DURATION = 5
N_CALIB_SAMPLES = 100  # representative dataset size for INT8 calibration

os.makedirs(TFLITE_DIR, exist_ok=True)


class MidStudent(nn.Module):
    def __init__(self, n_classes=50):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class TinyStudent(nn.Module):
    def __init__(self, n_classes=50):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def load_wav_to_mel(path):
    y, _ = librosa.load(path, sr=SR, duration=DURATION)
    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def export_onnx(model, name, input_shape=(1, 1, N_MELS, 216)):
    model.eval()
    dummy = torch.randn(*input_shape)
    onnx_path = os.path.join(TFLITE_DIR, f"{name}.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes=None,
    )
    return onnx_path


def representative_dataset_gen(calib_files):
    for path in calib_files:
        mel = load_wav_to_mel(path)
        x = mel[np.newaxis, :, :, np.newaxis]  # NHWC: [1, n_mels, time, 1]
        yield [x]


def convert_to_tflite_int8(onnx_path, name, calib_files):
    import subprocess
    out_dir = os.path.join(TFLITE_DIR, f"{name}_out")

    # plain onnx2tf conversion (no -oiqt): produces a SavedModel we quantize
    # ourselves, since onnx2tf's own calibration flags fought our value range.
    subprocess.run(
        ["onnx2tf", "-i", onnx_path, "-o", out_dir],
        check=True,
        timeout=300,
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(out_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(calib_files)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path = os.path.join(TFLITE_DIR, f"{name}_int8.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return out_path


def eval_tflite(tflite_path, val_df, mel_scale=127.0):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_scale, in_zero = input_details[0]["quantization"]

    correct, total = 0, 0
    for _, row in val_df.iterrows():
        path = os.path.join(AUDIO_DIR, row["filename"])
        mel = load_wav_to_mel(path)
        x = mel[np.newaxis, :, :, np.newaxis]  # NHWC
        x_q = (x / in_scale + in_zero).astype(np.int8) if in_scale else x.astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        pred = np.argmax(out)

        correct += int(pred == row["target"])
        total += 1
    return correct / total


def main():
    df = pd.read_csv(META_CSV)
    val_df = df[df["fold"] == 5]
    calib_files = [os.path.join(AUDIO_DIR, f) for f in df[df["fold"] != 5]["filename"].sample(N_CALIB_SAMPLES, random_state=0)]

    for name, model_class, ckpt in [
        ("mid_student", MidStudent, "student_mid_best.pt"),
        ("tiny_student", TinyStudent, "student_distilled_best.pt"),
    ]:
        print(f"\n=== {name} ===")
        model = model_class(n_classes=50)
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt), map_location="cpu"))

        onnx_path = export_onnx(model, name)
        print(f"exported ONNX: {onnx_path}")

        tflite_path = convert_to_tflite_int8(onnx_path, name, calib_files)
        tflite_size_kb = os.path.getsize(tflite_path) / 1024
        print(f"TFLite INT8 size: {tflite_size_kb:.1f} KB")

        acc = eval_tflite(tflite_path, val_df)
        print(f"TFLite INT8 val acc: {acc:.3f}")


if __name__ == "__main__":
    main()
