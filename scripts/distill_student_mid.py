"""
Milestone 3b: mid-size student, distilled from teacher_v2.
Target: middle point on the Pareto curve between tiny student and teacher.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

ESC50_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ESC-50")
AUDIO_DIR = os.path.join(ESC50_ROOT, "audio")
META_CSV = os.path.join(ESC50_ROOT, "meta/esc50.csv")
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
SR = 22050
N_MELS = 64
DURATION = 5
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
TEMPERATURE = 4.0
ALPHA = 0.7
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class ESC50Dataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

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
        if self.augment:
            mel_db = self._spec_augment(mel_db)
        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        return x, int(row["target"])

    def _spec_augment(self, mel):
        mel = mel.copy()
        f = np.random.randint(0, 10)
        f0 = np.random.randint(0, max(1, mel.shape[0] - f))
        mel[f0:f0 + f, :] = 0
        t = np.random.randint(0, 25)
        t0 = np.random.randint(0, max(1, mel.shape[1] - t))
        mel[:, t0:t0 + t] = 0
        return mel


class CNNTeacher(nn.Module):
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
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class MidStudent(nn.Module):
    """~8-15x compression target, sits between TinyStudent and teacher."""
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


def distill_loss(student_logits, teacher_logits, labels, T=TEMPERATURE, alpha=ALPHA):
    soft_teacher = F.log_softmax(student_logits / T, dim=1)
    soft_labels = F.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(soft_teacher, soft_labels, reduction="batchmean") * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * hard_loss


def run_epoch(student, teacher, loader, optimizer=None):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = distill_loss(student_logits, teacher_logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (student_logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    wandb.init(project="edge-ml-esc50", name="milestone3b-mid-student")

    df = pd.read_csv(META_CSV)
    train_df = df[df["fold"] != 5]
    val_df = df[df["fold"] == 5]
    train_ds = ESC50Dataset(train_df, augment=True)
    val_ds = ESC50Dataset(val_df, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    teacher = CNNTeacher(n_classes=50).to(DEVICE)
    teacher.load_state_dict(torch.load(os.path.join(CKPT_DIR, "teacher_v2_best.pt"), map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = MidStudent(n_classes=50).to(DEVICE)
    print(f"teacher params: {count_params(teacher):,}")
    print(f"student params: {count_params(student):,}")
    print(f"compression ratio: {count_params(teacher) / count_params(student):.1f}x")

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(student, teacher, train_loader, optimizer)
        val_loss, val_acc = run_epoch(student, teacher, val_loader, optimizer=None)
        scheduler.step()
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch+1}/{EPOCHS} | train_loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.3f} acc {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), os.path.join(CKPT_DIR, "student_mid_best.pt"))

    print(f"best mid-student val acc: {best_val_acc:.3f}")
    wandb.finish()


if __name__ == "__main__":
    main()
