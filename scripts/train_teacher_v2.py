"""
Milestone 2: push teacher accuracy with mixup + wider model + longer schedule.
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

ESC50_ROOT = os.path.expanduser("~/ESC-50")
AUDIO_DIR = os.path.join(ESC50_ROOT, "audio")
META_CSV = os.path.join(ESC50_ROOT, "meta/esc50.csv")
SR = 22050
N_MELS = 64
DURATION = 5
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
MIXUP_ALPHA = 0.3
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
        label = int(row["target"])
        return x, label

    def _spec_augment(self, mel):
        mel = mel.copy()
        f = np.random.randint(0, 10)
        f0 = np.random.randint(0, max(1, mel.shape[0] - f))
        mel[f0:f0 + f, :] = 0
        t = np.random.randint(0, 25)
        t0 = np.random.randint(0, max(1, mel.shape[1] - t))
        mel[:, t0:t0 + t] = 0
        return mel


def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


class CNNTeacher(nn.Module):
    """Wider than milestone 1: 32->64->128->256 channels."""
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


def run_train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_mixed, y_a, y_b, lam = mixup(x, y)

        optimizer.zero_grad()
        out = model(x_mixed)
        loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        # approximate train acc using dominant label
        preds = out.argmax(1)
        correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def run_val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def main():
    wandb.init(project="edge-ml-esc50", name="milestone2-mixup-wider")

    df = pd.read_csv(META_CSV)
    train_df = df[df["fold"] != 5]
    val_df = df[df["fold"] == 5]

    train_ds = ESC50Dataset(train_df, augment=True)
    val_ds = ESC50Dataset(val_df, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = CNNTeacher(n_classes=50).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_val_epoch(model, val_loader, criterion)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })
        print(f"epoch {epoch+1}/{EPOCHS} | train_loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.3f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.expanduser("~/edge-ml-esc50/checkpoints/teacher_v2_best.pt"))

    print(f"best val acc: {best_val_acc:.3f}")
    wandb.finish()


if __name__ == "__main__":
    main()
