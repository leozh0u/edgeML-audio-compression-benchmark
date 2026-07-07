#pragma once
// Portable iterative radix-2 complex FFT. Pure C, no esp-dsp/hardware
// dependency, so the exact same code runs on-device and in the host test
// harness (esp32/host_test/) for A/B validation against librosa.
//
// esp-dsp's dsps_fft2r_fc32 does the same radix-2 DIT math with
// hardware-accelerated (SIMD/assembly) butterflies; swapping it in later is a
// pure speed optimization, not a correctness change, since both compute the
// textbook radix-2 Cooley-Tukey FFT.

#ifdef __cplusplus
extern "C" {
#endif

// n must be a power of 2. In-place: re/im are length n, overwritten with the
// FFT of (re + i*im).
void fft_radix2(float* re, float* im, int n);

#ifdef __cplusplus
}
#endif
