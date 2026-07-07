#pragma once
#include "config.h"

// On-device log-mel spectrogram front end. Must reproduce, bit-for-bit-close,
// what librosa.feature.melspectrogram + power_to_db + per-clip z-norm produce in
// scripts/common.py::wav_to_mel — otherwise the model sees a different input
// distribution than it trained on and accuracy collapses.

void mel_frontend_init();

// pcm: CLIP_SAMPLES int16 samples @ SAMPLE_RATE.
// out: N_MELS * N_FRAMES floats, row-major [mel, frame], normalized (mean0/var1).
void mel_frontend_compute(const int16_t* pcm, float* out);
