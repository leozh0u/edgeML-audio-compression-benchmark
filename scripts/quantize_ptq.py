"""
Milestone 3c: post-training INT8 quantization on mid and tiny students.
Converts to TorchScript, applies dynamic quantization, measures size + accuracy.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ESC50_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ESC-50")
AUDIO_DIR = os.path.join(ESC50_ROOT, "audio")
META_CSV = os.path.join(ESC50_ROOT, "meta/esc50.csv")
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
SR = 22050
N_MELS = 64
DURATION = 5
BATCH_SIZE = 32
DEVICE = "cpu"  # quantized models run on CPU
torch.backends.quantized.engine = "qnnpack"  # required on Apple Silicon / ARM


class ESC50Dataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(AUDIO_DIR, row["filename"])
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        return x, int(row["target"])


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


def eval_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total


def model_size_kb(model, path="/tmp/tmp_model.pt"):
    torch.save(model.state_dict(), path)
    size_kb = os.path.getsize(path) / 1024
    os.remove(path)
    return size_kb


def quantize_and_eval(model_class, ckpt_name, label, val_loader):
    model = model_class(n_classes=50)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location="cpu"))
    model.eval()

    fp32_acc = eval_model(model, val_loader)
    fp32_size = model_size_kb(model)

    # dynamic quantization: quantizes Linear + Conv weights to int8
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    int8_acc = eval_model(quantized, val_loader)
    int8_size = model_size_kb(quantized)

    print(f"\n--- {label} ---")
    print(f"FP32: {fp32_acc:.3f} acc, {fp32_size:.1f} KB")
    print(f"INT8: {int8_acc:.3f} acc, {int8_size:.1f} KB")
    print(f"size reduction: {fp32_size / int8_size:.2f}x")
    print(f"accuracy delta: {int8_acc - fp32_acc:+.3f}")

    torch.save(quantized.state_dict(), os.path.join(CKPT_DIR, f"{label}_int8.pt"))
    return {
        "label": label, "fp32_acc": fp32_acc, "fp32_size_kb": fp32_size,
        "int8_acc": int8_acc, "int8_size_kb": int8_size,
    }


def main():
    df = pd.read_csv(META_CSV)
    val_df = df[df["fold"] == 5]
    val_ds = ESC50Dataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = []
    results.append(quantize_and_eval(MidStudent, "student_mid_best.pt", "mid_student", val_loader))
    results.append(quantize_and_eval(TinyStudent, "student_distilled_best.pt", "tiny_student", val_loader))

    print("\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
