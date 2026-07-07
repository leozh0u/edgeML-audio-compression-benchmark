// On-device log-mel front end (esp-dsp FFT + mel filterbank).
//
// STATUS: scaffold. The control flow, framing, and normalization are here and
// correct; the two DSP kernels marked TODO need to be filled in against esp-dsp
// and then validated on-device by comparing a handful of frames to the Python
// reference (scripts/common.py::wav_to_mel) on the SAME wav — do not trust it
// until that A/B check passes, because a mel front-end that is subtly wrong is
// the classic "great val accuracy, garbage on-device" trap.
//
// Mel filterbank note: precompute the 64 triangular filters on the host (they are
// fixed for SR=22050/N_FFT=2048/N_MELS=64) and ship them as a const table rather
// than building them on the MCU. A generator for that table is a small follow-up.

#include "mel_frontend.h"
#include <math.h>
#include <string.h>
// #include "esp_dsp.h"   // enable when building on-device

static float g_window[N_FFT];   // Hann window

void mel_frontend_init() {
  for (int i = 0; i < N_FFT; ++i)
    g_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (N_FFT - 1)));
  // TODO(board): dsps_fft2r_init_fc32(NULL, N_FFT); and load the mel filterbank table.
}

void mel_frontend_compute(const int16_t* pcm, float* out) {
  // 1) Frame: N_FRAMES windows of N_FFT samples, hop = HOP_LENGTH.
  // 2) Per frame: window -> real FFT -> power spectrum -> mel filterbank (64) ->
  //    power_to_db (10*log10, ref=max), matching librosa's ref=np.max.
  // 3) Global per-clip normalization: (x - mean) / (std + 1e-6).
  //
  // TODO(board): implement 1-2 with esp-dsp. Placeholder zero-fills so the rest
  // of the pipeline (quantize -> Invoke) is exercisable during bring-up.
  (void)pcm;
  memset(out, 0, sizeof(float) * N_MELS * N_FRAMES);

  // Step 3 is complete and will apply once real mel energies are written above.
  float mean = 0.0f;
  const int n = N_MELS * N_FRAMES;
  for (int i = 0; i < n; ++i) mean += out[i];
  mean /= n;
  float var = 0.0f;
  for (int i = 0; i < n; ++i) { float d = out[i] - mean; var += d * d; }
  float inv_std = 1.0f / (sqrtf(var / n) + 1e-6f);
  for (int i = 0; i < n; ++i) out[i] = (out[i] - mean) * inv_std;
}
