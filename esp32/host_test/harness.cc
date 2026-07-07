// Host-side (no ESP-IDF/board) driver for mel_frontend.{h,cc}. Lets the exact
// on-device DSP code be A/B-validated against librosa on a dev machine before
// ever touching hardware. See scripts/validate_mel_frontend.py.
//
// Usage: mel_frontend_test <pcm_s16le_path> <out_f32_path>
//   pcm_s16le_path: raw little-endian int16 mono PCM, exactly CLIP_SAMPLES samples.
//   out_f32_path:   written as N_MELS*N_FRAMES raw little-endian float32, row-major [mel, frame].

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../src/mel_frontend.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <pcm_s16le_path> <out_f32_path>\n", argv[0]);
    return 1;
  }

  std::vector<int16_t> pcm(CLIP_SAMPLES);
  FILE* fin = std::fopen(argv[1], "rb");
  if (!fin) { std::perror("open pcm"); return 1; }
  size_t got = std::fread(pcm.data(), sizeof(int16_t), CLIP_SAMPLES, fin);
  std::fclose(fin);
  if (got != (size_t)CLIP_SAMPLES) {
    std::fprintf(stderr, "expected %d int16 samples, got %zu\n", CLIP_SAMPLES, got);
    return 1;
  }

  mel_frontend_init();
  std::vector<float> mel(N_MELS * N_FRAMES);
  mel_frontend_compute(pcm.data(), mel.data());

  FILE* fout = std::fopen(argv[2], "wb");
  if (!fout) { std::perror("open out"); return 1; }
  std::fwrite(mel.data(), sizeof(float), mel.size(), fout);
  std::fclose(fout);
  return 0;
}
