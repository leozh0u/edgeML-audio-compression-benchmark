"""
Precompute mel-spectrogram features for all 2000 ESC-50 clips once and cache them
to feature_cache.npz. QAT and pruning fine-tune for many epochs; without this,
librosa re-decodes every wav every epoch and dominates runtime.

Run once (idempotent): python scripts/precompute_features.py
"""

import time
import numpy as np

import common as C


def main():
    import pandas as pd

    df = pd.read_csv(C.META_CSV).sort_values("filename").reset_index(drop=True)
    n = len(df)
    print(f"precomputing {n} mel features -> {C.CACHE_PATH}")

    feats = np.empty((n, C.N_MELS, 216), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    folds = np.empty(n, dtype=np.int64)

    t0 = time.time()
    for i, row in df.iterrows():
        mel = C.wav_to_mel(C.AUDIO_DIR / row["filename"])
        # Guard against off-by-one time-frame lengths across librosa versions.
        if mel.shape[1] != 216:
            fixed = np.zeros((C.N_MELS, 216), dtype=np.float32)
            w = min(mel.shape[1], 216)
            fixed[:, :w] = mel[:, :w]
            mel = fixed
        feats[i] = mel
        labels[i] = int(row["target"])
        folds[i] = int(row["fold"])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)")

    C.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.CACHE_PATH, feats=feats, labels=labels, folds=folds)
    mb = C.CACHE_PATH.stat().st_size / 1e6
    print(f"done: {C.CACHE_PATH.name} ({mb:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
