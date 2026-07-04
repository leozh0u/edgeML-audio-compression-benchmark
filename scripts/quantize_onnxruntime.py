"""
INT8 PTQ via ONNX Runtime static quantization.
Bypasses onnx2tf/TensorFlow entirely after repeated SavedModel-restore hangs.
Runs in tflite_env (has onnx already) or esc50env (has onnxruntime already) - either works.
"""

import os
import numpy as np
import pandas as pd
import librosa
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader

ESC50_ROOT = os.path.expanduser("~/ESC-50")
AUDIO_DIR = os.path.join(ESC50_ROOT, "audio")
META_CSV = os.path.join(ESC50_ROOT, "meta/esc50.csv")
TFLITE_DIR = os.path.expanduser("~/edge-ml-esc50/tflite_models")
SR = 22050
N_MELS = 64
DURATION = 5


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


class MelCalibrationReader(CalibrationDataReader):
    def __init__(self, calib_files, input_name):
        self.files = calib_files
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        mel = load_wav_to_mel(self.files[self.idx])
        x = mel[np.newaxis, np.newaxis, :, :]  # NCHW, matches original torch.onnx.export
        self.idx += 1
        return {self.input_name: x}


def eval_onnx(session, val_df):
    input_name = session.get_inputs()[0].name
    correct, total = 0, 0
    for _, row in val_df.iterrows():
        path = os.path.join(AUDIO_DIR, row["filename"])
        mel = load_wav_to_mel(path)
        x = mel[np.newaxis, np.newaxis, :, :]
        out = session.run(None, {input_name: x})[0]
        pred = np.argmax(out)
        correct += int(pred == row["target"])
        total += 1
    return correct / total


def quantize_and_eval(onnx_path, name, calib_files, val_df):
    fp32_size = os.path.getsize(onnx_path) / 1024
    fp32_session = ort.InferenceSession(onnx_path)
    fp32_acc = eval_onnx(fp32_session, val_df)

    input_name = fp32_session.get_inputs()[0].name
    int8_path = os.path.join(TFLITE_DIR, f"{name}_int8.onnx")

    quantize_static(
        model_input=onnx_path,
        model_output=int8_path,
        calibration_data_reader=MelCalibrationReader(calib_files, input_name),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    int8_size = os.path.getsize(int8_path) / 1024
    int8_session = ort.InferenceSession(int8_path)
    int8_acc = eval_onnx(int8_session, val_df)

    print(f"\n--- {name} ---")
    print(f"FP32: {fp32_acc:.3f} acc, {fp32_size:.1f} KB")
    print(f"INT8: {int8_acc:.3f} acc, {int8_size:.1f} KB")
    print(f"size reduction: {fp32_size / int8_size:.2f}x")
    print(f"accuracy delta: {int8_acc - fp32_acc:+.3f}")
    return {"name": name, "fp32_acc": fp32_acc, "fp32_kb": fp32_size,
            "int8_acc": int8_acc, "int8_kb": int8_size}


def main():
    df = pd.read_csv(META_CSV)
    val_df = df[df["fold"] == 5]
    calib_files = [os.path.join(AUDIO_DIR, f) for f in
                   df[df["fold"] != 5]["filename"].sample(50, random_state=0)]

    results = []
    for name in ["mid_student", "tiny_student"]:
        onnx_path = os.path.join(TFLITE_DIR, f"{name}.onnx")
        if not os.path.exists(onnx_path):
            print(f"skip {name}: {onnx_path} not found, export ONNX first")
            continue
        results.append(quantize_and_eval(onnx_path, name, calib_files, val_df))

    print("\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
