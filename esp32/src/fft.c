#include "fft.h"
#include <math.h>

static int bit_reverse(int x, int log2n) {
  int r = 0;
  for (int i = 0; i < log2n; ++i) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

void fft_radix2(float* re, float* im, int n) {
  int log2n = 0;
  while ((1 << log2n) < n) ++log2n;

  for (int i = 0; i < n; ++i) {
    int j = bit_reverse(i, log2n);
    if (j > i) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }

  for (int s = 1; s <= log2n; ++s) {
    int m = 1 << s;
    int half = m >> 1;
    float theta = -2.0f * (float)M_PI / (float)m;
    float wm_re = cosf(theta), wm_im = sinf(theta);
    for (int k = 0; k < n; k += m) {
      float w_re = 1.0f, w_im = 0.0f;
      for (int j = 0; j < half; ++j) {
        int a = k + j, b = k + j + half;
        float t_re = w_re * re[b] - w_im * im[b];
        float t_im = w_re * im[b] + w_im * re[b];
        re[b] = re[a] - t_re;
        im[b] = im[a] - t_im;
        re[a] = re[a] + t_re;
        im[a] = im[a] + t_im;
        float nw_re = w_re * wm_re - w_im * wm_im;
        float nw_im = w_re * wm_im + w_im * wm_re;
        w_re = nw_re; w_im = nw_im;
      }
    }
  }
}
