"""
Shared building blocks for the edge-ML ESC-50 pipeline.

Everything reusable lives here so the training / distillation / QAT / pruning /
benchmark scripts agree on ONE definition of paths, feature extraction, the
dataset, and the model architectures. The architectures below must stay byte-for-byte
compatible with the saved checkpoints (exact layer specs in DECISIONS.md).

Path resolution is location-independent: it derives everything from this file's
location, so it works whether the repo is at ~/edge-ml-esc50 (old handoff layout)
or ~/Projects/edgeml-audio/edge-ml-esc50 (current layout). The old scripts
hardcoded ~/ESC-50 and ~/edge-ml-esc50 and silently broke after the move.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file, with sensible fallbacks)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent          # .../edge-ml-esc50
CKPT_DIR = REPO_ROOT / "checkpoints"
TFLITE_DIR = REPO_ROOT / "tflite_models"
CACHE_PATH = REPO_ROOT / "feature_cache.npz"                 # precomputed mels (gitignored)
RESULTS_DIR = REPO_ROOT / "results"


def _find_dataset_root() -> Path:
    """ESC-50 is a sibling clone of the repo; fall back to the old ~/ location."""
    candidates = [
        REPO_ROOT.parent / "ESC-50",   # current layout: alongside the repo
        Path.home() / "ESC-50",        # original handoff layout
        REPO_ROOT / "ESC-50",          # if someone nests it inside the repo
    ]
    for c in candidates:
        if (c / "meta" / "esc50.csv").exists():
            return c
    # Return the most likely path anyway so the error message is actionable.
    return candidates[0]


ESC50_ROOT = _find_dataset_root()
AUDIO_DIR = ESC50_ROOT / "audio"
META_CSV = ESC50_ROOT / "meta" / "esc50.csv"

# --------------------------------------------------------------------------- #
# Audio / feature config (must match how the checkpoints were trained)
# --------------------------------------------------------------------------- #
SR = 22050
N_MELS = 64
DURATION = 5
TARGET_LEN = SR * DURATION
INPUT_SHAPE = (1, N_MELS, 216)   # (C, mel bins, time frames) after 5s @ 22050


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def wav_to_mel(path) -> np.ndarray:
    """5s wav -> normalized log-mel spectrogram, shape (N_MELS, ~216), float32."""
    y, _ = librosa.load(str(path), sr=SR, duration=DURATION)
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


class ESC50Dataset(Dataset):
    """Standard ESC-50 folds. augment=True applies SpecAugment (train only)."""

    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = wav_to_mel(AUDIO_DIR / row["filename"])
        if self.augment:
            mel = self._spec_augment(mel)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return x, int(row["target"])

    @staticmethod
    def _spec_augment(mel):
        mel = mel.copy()
        f = np.random.randint(0, 10)
        f0 = np.random.randint(0, max(1, mel.shape[0] - f))
        mel[f0:f0 + f, :] = 0
        t = np.random.randint(0, 25)
        t0 = np.random.randint(0, max(1, mel.shape[1] - t))
        mel[:, t0:t0 + t] = 0
        return mel


def load_folds():
    """Return (train_df, val_df) using the standard folds-1-4 / fold-5 split."""
    df = pd.read_csv(META_CSV)
    return df[df["fold"] != 5], df[df["fold"] == 5]


class CachedESC50(Dataset):
    """
    In-memory dataset backed by precomputed mel features (see precompute_features.py).

    Avoids re-running librosa every epoch, which is the dominant cost for the
    iterative QAT / pruning fine-tuning loops. SpecAugment is still applied
    on-the-fly at train time so augmentation stays stochastic.
    """

    def __init__(self, feats, labels, augment=False):
        self.feats = feats            # float32 array [N, N_MELS, T]
        self.labels = labels          # int array [N]
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.feats[idx]
        if self.augment:
            mel = ESC50Dataset._spec_augment(mel)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return x, int(self.labels[idx])


def load_cached_folds():
    """
    Load precomputed features and return (train_ds, val_ds) as CachedESC50.
    Raises a clear error if the cache is missing.
    """
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Feature cache not found at {CACHE_PATH}. "
            f"Run:  python scripts/precompute_features.py"
        )
    data = np.load(CACHE_PATH)
    feats, labels, folds = data["feats"], data["labels"], data["folds"]
    tr = folds != 5
    va = folds == 5
    return (
        CachedESC50(feats[tr], labels[tr], augment=True),
        CachedESC50(feats[va], labels[va], augment=False),
    )


# --------------------------------------------------------------------------- #
# Model architectures (frozen; must match committed checkpoints)
# --------------------------------------------------------------------------- #
class CNNTeacher(nn.Module):
    """Milestone-2 teacher. 992,242 params. -> teacher_v2_best.pt"""

    def __init__(self, n_classes=50):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(self.dropout(x))


class MidStudent(nn.Module):
    """63,826 params, 15.5x compression. -> student_mid_best.pt"""

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
        return self.classifier(self.features(x).flatten(1))


class TinyStudent(nn.Module):
    """16,962 params, 58.5x compression. -> student_distilled_best.pt"""

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
        return self.classifier(self.features(x).flatten(1))


# Registry so downstream scripts can iterate over the student zoo uniformly.
STUDENTS = {
    "mid_student": {"cls": MidStudent, "ckpt": "student_mid_best.pt"},
    "tiny_student": {"cls": TinyStudent, "ckpt": "student_distilled_best.pt"},
}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def distill_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft_teacher = F.log_softmax(student_logits / T, dim=1)
    soft_labels = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(soft_teacher, soft_labels, reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1 - alpha) * hard


def load_teacher(dev=None):
    dev = dev or device()
    teacher = CNNTeacher().to(dev)
    teacher.load_state_dict(torch.load(CKPT_DIR / "teacher_v2_best.pt", map_location=dev))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def evaluate(model, loader, dev=None):
    dev = dev or device()
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


if __name__ == "__main__":
    # Quick self-check of paths + architectures.
    print("REPO_ROOT :", REPO_ROOT)
    print("ESC50_ROOT:", ESC50_ROOT, "(exists)" if META_CSV.exists() else "(MISSING meta/esc50.csv)")
    print("CKPT_DIR  :", CKPT_DIR)
    print("device    :", device())
    for name, spec in STUDENTS.items():
        m = spec["cls"]()
        print(f"{name:12} {count_params(m):>8,} params  ckpt={spec['ckpt']}")
    print(f"{'teacher':12} {count_params(CNNTeacher()):>8,} params")
