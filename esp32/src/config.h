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

// TFLite Micro tensor arena. 200KB fits the tiny model in fast internal SRAM
// (measured arena_used ~140KB) with headroom for WiFi/mic buffers. The mid model
// needs ~273KB and only fits PSRAM (slow) — see DECISIONS.md M9 tradeoff.
#define TENSOR_ARENA_KB  200

// I2S wiring for an INMP441 MEMS mic (optional live-inference input).
#define I2S_WS_PIN       15
#define I2S_SCK_PIN      14
#define I2S_SD_PIN       32

// Dashboard bridge (server.py runs the WebSocket on port 8765).
#define WIFI_SSID        "YOUR_SSID"
#define WIFI_PASS        "YOUR_PASS"
#define DASHBOARD_HOST   "192.168.1.100"   // machine running dashboard/server.py
#define DASHBOARD_WS_PORT 8765
