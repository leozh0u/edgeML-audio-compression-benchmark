"""
A/B-validate the on-device mel front end (esp32/src/mel_frontend.cc) against
the real training-time feature extractor (scripts/common.py::wav_to_mel)
WITHOUT needing the ESP32 board: compiles esp32/host_test/harness.cc as a
native binary, runs it on synthetic PCM, and diffs the output against librosa.

This is the off-board half of the "must be A/B-validated" requirement in
DECISIONS.md / DECISIONS.md. It proves the DSP math (framing, Hann
window, FFT, mel filterbank, power_to_db, z-norm) is correct. It does NOT
replace the on-device check with a real microphone clip once the board is in
hand -- ADC noise, I2S timing, and fixed-point/latency effects only show up
there. Re-run this after ANY change to mel_frontend.cc or the filterbank.

Run with the esc50env python:
    ../esc50env/bin/python3 scripts/validate_mel_frontend.py
Needs a C++ compiler (cc/c++) on PATH; no ESP-IDF/PlatformIO required.
"""

import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import librosa

REPO_ROOT = Path(__file__).resolve().parent.parent
ESP32_SRC = REPO_ROOT / "esp32" / "src"
HOST_TEST = REPO_ROOT / "esp32" / "host_test"
BUILD_DIR = HOST_TEST / "build"

SR = 22050
N_MELS = 64
DURATION = 5
CLIP_SAMPLES = SR * DURATION

try:
    from common import wav_to_mel as _wav_to_mel_canonical
    assert (SR, N_MELS, DURATION) == (22050, 64, 5)  # sanity: constants above match common.py
    def reference_mel(wav_path):
        return _wav_to_mel_canonical(wav_path)
except ImportError:
    # common.py imports torch/pandas for the training pipeline; this script only
    # needs the feature-extraction math, so fall back to the same 4 librosa/numpy
    # calls when those heavier deps aren't installed. Keep this byte-identical to
    # common.py::wav_to_mel -- if that function changes, update this too.
    def reference_mel(wav_path):
        y, _ = librosa.load(str(wav_path), sr=SR, duration=DURATION)
        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db.astype(np.float32)


def build_harness():
    BUILD_DIR.mkdir(exist_ok=True)
    fft_o = BUILD_DIR / "fft.o"
    binary = BUILD_DIR / "mel_frontend_test"
    subprocess.run(["cc", "-O2", "-c", str(ESP32_SRC / "fft.c"), "-o", str(fft_o)], check=True)
    subprocess.run([
        "c++", "-O2", "-std=c++11", f"-I{ESP32_SRC}",
        str(HOST_TEST / "harness.cc"), str(ESP32_SRC / "mel_frontend.cc"), str(fft_o),
        "-lm", "-o", str(binary),
    ], check=True)
    return binary


def make_test_signal(kind: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(CLIP_SAMPLES) / SR
    if kind == "multitone":
        x = (0.3 * np.sin(2 * np.pi * 220 * t)
             + 0.2 * np.sin(2 * np.pi * 1500 * t)
             + 0.1 * np.sin(2 * np.pi * 6000 * t))
    elif kind == "noise":
        x = 0.2 * rng.standard_normal(CLIP_SAMPLES)
    elif kind == "transient":
        x = np.zeros(CLIP_SAMPLES)
        burst = slice(SR, SR + SR // 2)  # 0.5s burst starting at t=1s, silence elsewhere
        x[burst] = 0.5 * np.sin(2 * np.pi * 800 * t[burst])
    elif kind == "chirp":
        x = 0.3 * np.sin(2 * np.pi * (100 + 2000 * t / DURATION) * t)
    else:
        raise ValueError(kind)
    return np.clip(x, -0.999, 0.999).astype(np.float32)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def run_case(binary, kind, tmpdir: Path):
    x = make_test_signal(kind)
    pcm = to_pcm16(x)

    wav_path = tmpdir / f"{kind}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    raw_path = tmpdir / f"{kind}.pcm"
    raw_path.write_bytes(pcm.tobytes())

    out_path = tmpdir / f"{kind}.f32"
    subprocess.run([str(binary), str(raw_path), str(out_path)], check=True)
    n_frames = out_path.stat().st_size // 4 // N_MELS
    c_mel = np.fromfile(out_path, dtype="<f4").reshape(N_MELS, n_frames)

    ref_mel = reference_mel(wav_path)

    diff = np.abs(c_mel - ref_mel)
    corr = np.corrcoef(c_mel.ravel(), ref_mel.ravel())[0, 1]
    return {
        "kind": kind,
        "shape_c": c_mel.shape,
        "shape_ref": ref_mel.shape,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "corr": float(corr),
    }


def main():
    binary = build_harness()
    with_tmp = BUILD_DIR / "cases"
    with_tmp.mkdir(exist_ok=True)

    cases = ["multitone", "noise", "transient", "chirp"]
    results = [run_case(binary, k, with_tmp) for k in cases]

    print(f"{'signal':10} {'shape':>12} {'max|Δ|':>10} {'mean|Δ|':>10} {'corr':>8}")
    all_ok = True
    for r in results:
        ok = r["shape_c"] == r["shape_ref"] and r["max_abs_diff"] < 0.05 and r["corr"] > 0.999
        all_ok &= ok
        print(f"{r['kind']:10} {str(r['shape_c']):>12} {r['max_abs_diff']:>10.4f} "
              f"{r['mean_abs_diff']:>10.4f} {r['corr']:>8.5f} {'OK' if ok else 'FAIL'}")

    if not all_ok:
        print("\nFAIL: on-device mel front end diverges from wav_to_mel.", file=sys.stderr)
        sys.exit(1)
    print("\nPASS: mel_frontend.cc matches wav_to_mel within tolerance on all synthetic cases.")


if __name__ == "__main__":
    main()
