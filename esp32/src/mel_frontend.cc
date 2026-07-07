// On-device log-mel front end. Reproduces scripts/common.py::wav_to_mel
// (librosa.feature.melspectrogram + power_to_db(ref=np.max) + per-clip
// z-norm) using a portable radix-2 FFT (fft.c) and a mel filterbank table
// precomputed on the host by scripts/gen_mel_filterbank.py.
//
// Validated off-board against the real librosa pipeline via
// esp32/host_test/ + scripts/validate_mel_frontend.py — see DECISIONS.md.
// esp-dsp is not used: fft_radix2 does the same textbook radix-2 DIT math
// esp-dsp's dsps_fft2r_fc32 does, just without hardware-specific SIMD/asm.
// Swapping it in later (dsps_fft2r_fc32 + dsps_cplx2real_fc32) is a pure
// speed optimization once on the board, not a correctness change.

#include "mel_frontend.h"
#include "fft.h"
#include "mel_filterbank_data.h"
#include <math.h>
#include <string.h>

static float g_window[N_FFT];        // periodic Hann (matches librosa fftbins=True)
static const int N_BINS = N_FFT / 2 + 1;

void mel_frontend_init() {
  for (int i = 0; i < N_FFT; ++i)
    g_window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / N_FFT);
}

void mel_frontend_compute(const int16_t* pcm, float* out) {
  static float frame_re[N_FFT];
  static float frame_im[N_FFT];
  static float power[N_FFT / 2 + 1];

  const int pad = N_FFT / 2;   // librosa center=True, pad_mode='constant' (zero)

  // 1-2) Frame -> window -> FFT -> power spectrum -> mel filterbank, per frame.
  // out is row-major [mel, frame]: out[m * N_FRAMES + f].
  for (int f = 0; f < N_FRAMES; ++f) {
    int frame_start = f * HOP_LENGTH - pad;
    for (int i = 0; i < N_FFT; ++i) {
      int idx = frame_start + i;
      float sample = (idx >= 0 && idx < CLIP_SAMPLES) ? (pcm[idx] / 32768.0f) : 0.0f;
      frame_re[i] = sample * g_window[i];
      frame_im[i] = 0.0f;
    }

    fft_radix2(frame_re, frame_im, N_FFT);

    for (int k = 0; k < N_BINS; ++k)
      power[k] = frame_re[k] * frame_re[k] + frame_im[k] * frame_im[k];

    for (int m = 0; m < N_MELS; ++m) {
      const float* w = &MEL_WEIGHTS[MEL_WEIGHT_OFFSET[m]];
      const uint16_t start = MEL_START_BIN[m];
      const uint16_t count = MEL_NUM_BINS[m];
      float e = 0.0f;
      for (int b = 0; b < count; ++b)
        e += w[b] * power[start + b];
      out[m * N_FRAMES + f] = e;
    }
  }

  // 3) power_to_db(ref=np.max, top_db=80), matching librosa.power_to_db exactly.
  const int n = N_MELS * N_FRAMES;
  const float amin = 1e-10f;

  float max_power = 0.0f;
  for (int i = 0; i < n; ++i) if (out[i] > max_power) max_power = out[i];
  float ref_db = 10.0f * log10f(fmaxf(amin, max_power));

  float max_db = -1e30f;
  for (int i = 0; i < n; ++i) {
    float db = 10.0f * log10f(fmaxf(amin, out[i])) - ref_db;
    out[i] = db;
    if (db > max_db) max_db = db;
  }
  const float floor_db = max_db - 80.0f;
  for (int i = 0; i < n; ++i) if (out[i] < floor_db) out[i] = floor_db;

  // 4) Per-clip normalization: (x - mean) / (std + 1e-6), matches wav_to_mel.
  float mean = 0.0f;
  for (int i = 0; i < n; ++i) mean += out[i];
  mean /= n;
  float var = 0.0f;
  for (int i = 0; i < n; ++i) { float d = out[i] - mean; var += d * d; }
  float inv_std = 1.0f / (sqrtf(var / n) + 1e-6f);
  for (int i = 0; i < n; ++i) out[i] = (out[i] - mean) * inv_std;
}
