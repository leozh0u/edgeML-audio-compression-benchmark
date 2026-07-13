#pragma once
// Single source of truth for the audio + model front-end parameters.
// These MUST match how the model was trained (see scripts/common.py):
//   SR=22050, N_MELS=64, DURATION=5s, 216 time frames, per-clip mean/var norm.
// If any of these drift from the training config, on-device accuracy collapses
// silently — that is the #1 deployment footgun, so they live in one place.

#define SAMPLE_RATE      22050
#define CLIP_SECONDS     5
#define CLIP_SAMPLES     (SAMPLE_RATE * CLIP_SECONDS)   // 110250
#define N_MELS           64
#define N_FRAMES         216          // librosa default hop (512) over 5s @ 22050
#define N_FFT            2048
#define HOP_LENGTH       512
#define N_CLASSES        50

// TFLite Micro tensor arena. Deployed model is mid int8 (70%): measured arena_used
// ~273KB, which exceeds internal SRAM so it lands in PSRAM — fine now that esp-nn
// SIMD kernels bring inference to ~416ms (was 22.5s with reference kernels; see
// DECISIONS.md M9). 320KB gives headroom. For the low-latency tiny model (143ms,
// 54%) drop this to 200 and regen model_data.cc from the tiny tflite.
#define TENSOR_ARENA_KB  320

// I2S wiring for an INMP441 MEMS mic (optional live-inference input).
// NOTE: pins must be free GPIOs on the ESP32-S3-WROOM-1 (N16R8). GPIO26-37 are
// bonded to the in-package SPI flash + octal PSRAM and are NOT usable — the old
// SD=32 (a classic-ESP32 I2S pin) is a flash data line on the S3. 4/5/6 are free.
#define I2S_SCK_PIN      4     // BCLK (bit clock)
#define I2S_WS_PIN       5     // LRCLK / word select
#define I2S_SD_PIN       6     // data out from mic

// Dashboard bridge (server.py runs the WebSocket on port 8765).
#define WIFI_SSID        "YOUR_SSID"
#define WIFI_PASS        "YOUR_PASS"
#define DASHBOARD_HOST   "192.168.1.100"   // machine running dashboard/server.py
#define DASHBOARD_WS_PORT 8765
