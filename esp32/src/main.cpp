// ESC-50 edge classifier — ESP32-S3 firmware.
//
// Pipeline:  I2S mic (INMP441) -> 5s buffer -> log-mel spectrogram (mel_frontend.cc:
//            FFT + mel filterbank) -> TFLite Micro INT8 inference -> top-1 class
//            -> push {label, confidence, latency} to the dashboard over WebSocket.
//
// Three build-time modes (set -DBRINGUP_MODE=N in platformio.ini or the CLI):
//   1 = MODELCHECK : run inference on a zero input. Proves the graph loads/allocates
//                    and gives baseline latency + arena RAM. No mel, no mic. Flash
//                    this FIRST on a new board.
//   2 = SELFTEST   : run the full mel_frontend -> inference pipeline on real ESC-50
//                    clips embedded in flash (test_clips.h). Proves the on-device DSP
//                    preserves predictions on real audio WITHOUT needing the mic wired
//                    — the on-device half of the mel A/B. This is the default.
//   0 = MIC        : capture live audio from the INMP441 over I2S, classify, and push
//                    to the dashboard. Requires the mic physically wired (see config.h
//                    I2S_* pins) and on-board gain calibration — that is the one step
//                    the host validation cannot cover. Enable after SELFTEST passes.
//
// int8 quant params (input/output scale + zero-point) are read from the .tflite at
// runtime via tensor->params, never hardcoded.

#include <Arduino.h>
#include "esp_heap_caps.h"
#include "config.h"
#include "mel_frontend.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esc50_model/model_data.h"
// The old tanakamasayuki lib uses the pre-ErrorReporter-removal TFLite Micro API;
// the esp-nn build (ESP_TF / esp-tflite-micro) uses the modern one that dropped it.
#ifndef ESP_NN
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif

#ifndef BRINGUP_MODE
#define BRINGUP_MODE 2          // default: on-device self-test on embedded real clips
#endif
#define MODE_MIC        0
#define MODE_MODELCHECK 1
#define MODE_SELFTEST   2

#if BRINGUP_MODE == MODE_SELFTEST
#include "test_clips.h"
#endif
#if BRINGUP_MODE == MODE_MIC
#include "driver/i2s.h"
#endif

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
// The mid model's peak arena (~216KB first-conv feature map) does not fit the
// ESP32-S3's ~320KB internal DRAM alongside the framework/wifi/.bss, so the arena
// is allocated from the 8MB PSRAM at runtime (falls back to internal DRAM if PSRAM
// is absent — which only the tiny model would fit). Trades some latency (PSRAM is
// slower than internal SRAM) for keeping the higher-accuracy mid model on-device.
const size_t kArenaSize = (size_t)TENSOR_ARENA_KB * 1024;
uint8_t* tensor_arena = nullptr;

// 50 ESC-50 class names, index-aligned with meta/esc50.csv targets.
const char* CLASSES[N_CLASSES] = {
  "dog","rooster","pig","cow","frog","cat","hen","insects","sheep","crow",
  "rain","sea_waves","crackling_fire","crickets","chirping_birds","water_drops",
  "wind","pouring_water","toilet_flush","thunderstorm","crying_baby","sneezing",
  "clapping","breathing","coughing","footsteps","laughing","brushing_teeth",
  "snoring","drinking_sipping","door_wood_knock","mouse_click","keyboard_typing",
  "door_wood_creaks","can_opening","washing_machine","vacuum_cleaner","clock_alarm",
  "clock_tick","glass_breaking","helicopter","chainsaw","siren","car_horn",
  "engine","train","church_bells","airplane","fireworks","hand_saw"
};

float mel_buf[N_MELS * N_FRAMES];   // normalized log-mel, model input (float view)
#if BRINGUP_MODE == MODE_MIC
int16_t* pcm_buf = nullptr;         // 5s capture buffer, allocated in PSRAM (220KB)
#endif
}  // namespace

// Quantize a float mel-spectrogram into the model's int8 input tensor using the
// scale/zero-point baked into the .tflite (never hardcode these).
static void fill_input_int8(const float* mel) {
  const float scale = input->params.scale;
  const int zp = input->params.zero_point;
  int8_t* dst = input->data.int8;
  for (int i = 0; i < N_MELS * N_FRAMES; ++i) {
    int v = lroundf(mel[i] / scale) + zp;
    dst[i] = (int8_t)constrain(v, -128, 127);
  }
}

// Run inference on a prepared float mel. Returns top-1 index; fills confidence
// (softmax prob) and latency (ms).
static int run_inference(const float* mel, float* out_conf, float* out_ms) {
  fill_input_int8(mel);
  uint32_t t0 = micros();
  if (interpreter->Invoke() != kTfLiteOk) { Serial.println("Invoke failed"); return -1; }
  uint32_t dt = micros() - t0;

  const float oscale = output->params.scale;
  const int ozp = output->params.zero_point;
  // top-1 + softmax confidence over the dequantized logits.
  float logits[N_CLASSES];
  float maxl = -1e30f;
  int best = 0;
  for (int c = 0; c < N_CLASSES; ++c) {
    logits[c] = (output->data.int8[c] - ozp) * oscale;
    if (logits[c] > maxl) { maxl = logits[c]; best = c; }
  }
  float sum = 0.0f;
  for (int c = 0; c < N_CLASSES; ++c) sum += expf(logits[c] - maxl);
  *out_conf = 1.0f / sum;                 // exp(max-max)/sum = softmax prob of best
  *out_ms = dt / 1000.0f;
  return best;
}

// Machine-readable line for the dashboard serial bridge (server.py --serial parses
// lines beginning "RESULT "). Kept separate from the human-readable logs above.
static void emit_result(int cls, float conf, float ms, const char* source) {
  if (cls < 0) return;
  Serial.printf("RESULT {\"label\":\"%s\",\"class_id\":%d,\"confidence\":%.3f,"
                "\"latency_ms\":%.1f,\"source\":\"%s\"}\n",
                CLASSES[cls], cls, conf, ms, source);
}

#if BRINGUP_MODE == MODE_MIC
// --- INMP441 I2S capture ------------------------------------------------------
// SCAFFOLD: compiles and runs, but the >>14 gain shift and clock timing WILL need
// on-board calibration against a known clip — that is the documented board-only
// risk step. Compare a captured clip's mel to common.wav_to_mel of the same source
// before trusting live predictions.
static void i2s_mic_init() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,   // INMP441 is a 24-bit-in-32 device
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0,
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN,
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr);
  i2s_set_pin(I2S_NUM_0, &pins);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

// Capture CLIP_SAMPLES into pcm_buf (int16). INMP441 packs its 24-bit sample in the
// high bits of a 32-bit word; the shift lands it in int16 range. >>15 (vs >>14)
// was chosen after measuring on-board: at >>14 loud transients peaked ~31661,
// clipping the int16 ceiling; >>15 gives ~16k peak headroom. The mel front end's
// power_to_db(ref=max) + per-clip z-norm make it gain-invariant, so the shift only
// has to avoid clipping while staying above the noise floor — both true at >>15.
static void i2s_capture(int16_t* dst) {
  static int32_t raw[512];
  int filled = 0;
  while (filled < CLIP_SAMPLES) {
    size_t nbytes = 0;
    i2s_read(I2S_NUM_0, raw, sizeof(raw), &nbytes, portMAX_DELAY);
    int n = nbytes / sizeof(int32_t);
    for (int i = 0; i < n && filled < CLIP_SAMPLES; ++i)
      dst[filled++] = (int16_t)(raw[i] >> 15);
  }
}
#endif

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.printf("\nESC-50 edge classifier booting (BRINGUP_MODE=%d)...\n", BRINGUP_MODE);

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("model schema %lu != supported %d\n",
                  (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Exactly the ops in mid_student_int8_clean.tflite (verified with the tflite
  // interpreter). ReLU is fused into CONV_2D; input is already int8 so there is
  // no standalone QUANTIZE op; GlobalAveragePool lowers to MEAN.
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();              // GlobalAveragePool2D -> MEAN
  resolver.AddFullyConnected();

  // Prefer internal SRAM (fast); fall back to PSRAM only if the arena won't fit.
  // This matters enormously: the mid model's 273KB arena only fits PSRAM, where
  // inference is ~22.5s (PSRAM bandwidth bound); the tiny model's arena fits
  // internal SRAM and runs ~100x faster. See DECISIONS.md M9.
  tensor_arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  bool arena_in_psram = false;
  if (!tensor_arena) {
    tensor_arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    arena_in_psram = (tensor_arena != nullptr);
  }
  if (!tensor_arena) { Serial.println("arena alloc failed"); return; }
  Serial.printf("arena: %u KB requested in %s\n", (unsigned)(kArenaSize / 1024),
                arena_in_psram ? "PSRAM" : "internal SRAM");

  // tanakamasayuki/TensorFlowLite_ESP32 tracks an older TFLite Micro API whose
  // MicroInterpreter still takes an ErrorReporter* (the modern esp-nn build drops it).
#ifdef ESP_NN
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize);
#else
  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize, &micro_error_reporter);
#endif
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed — raise TENSOR_ARENA_KB");
    return;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("model loaded. arena used: %u / %u bytes. input:[%d,%d,%d,%d] type=%d\n",
                (unsigned)interpreter->arena_used_bytes(), (unsigned)kArenaSize,
                input->dims->data[0], input->dims->data[1],
                input->dims->data[2], input->dims->data[3], input->type);
  Serial.printf("heap: free=%u  psram_free=%u\n",
                (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());

  mel_frontend_init();

#if BRINGUP_MODE == MODE_MIC
  pcm_buf = (int16_t*)heap_caps_malloc((size_t)CLIP_SAMPLES * sizeof(int16_t),
                                       MALLOC_CAP_SPIRAM);
  if (!pcm_buf) { Serial.println("PSRAM alloc for pcm_buf failed"); return; }
  i2s_mic_init();
  Serial.println("I2S mic initialized. Capturing live audio...");
  // wifi_bridge_init();  // enable once WIFI_* in config.h is set
#endif
}

void loop() {
#if BRINGUP_MODE == MODE_MODELCHECK
  for (int i = 0; i < N_MELS * N_FRAMES; ++i) mel_buf[i] = 0.0f;
  float conf, ms;
  int cls = run_inference(mel_buf, &conf, &ms);
  Serial.printf("[modelcheck] pred=%-14s conf=%.2f  latency=%.1fms\n",
                cls >= 0 ? CLASSES[cls] : "ERR", conf, ms);
  delay(1500);

#elif BRINGUP_MODE == MODE_SELFTEST
  int pass = 0;
  float ms_sum = 0.0f;
  for (int k = 0; k < N_TEST_CLIPS; ++k) {
    mel_frontend_compute(TEST_CLIP_PCM[k], mel_buf);
    float conf, ms;
    int cls = run_inference(mel_buf, &conf, &ms);
    int exp = TEST_CLIP_EXPECT[k];
    bool ok = (cls == exp);
    pass += ok;
    ms_sum += ms;
    Serial.printf("[selftest %d/%d] pred=%-14s conf=%.2f  expect=%-14s  %s  latency=%.1fms\n",
                  k + 1, N_TEST_CLIPS, cls >= 0 ? CLASSES[cls] : "ERR", conf,
                  CLASSES[exp], ok ? "PASS" : "FAIL", ms);
    emit_result(cls, conf, ms, "selftest");
  }
  Serial.printf("[selftest] %d/%d passed, mean inference latency=%.1fms\n\n",
                pass, N_TEST_CLIPS, ms_sum / N_TEST_CLIPS);
  delay(3000);

#else  // MODE_MIC
  i2s_capture(pcm_buf);
  // Gain diagnostic: peak/RMS of the captured int16 buffer. Target for a loud
  // sound is a peak in the low thousands to ~20000 (of int16's 32767 max); a
  // near-zero peak = gain too low / miswire, a pinned 32767 = saturating.
  int32_t peak = 0; double sumsq = 0.0;
  for (int i = 0; i < CLIP_SAMPLES; ++i) {
    int32_t a = pcm_buf[i] < 0 ? -pcm_buf[i] : pcm_buf[i];
    if (a > peak) peak = a;
    sumsq += (double)pcm_buf[i] * pcm_buf[i];
  }
  int rms = (int)sqrt(sumsq / CLIP_SAMPLES);
  uint32_t t0 = micros();
  mel_frontend_compute(pcm_buf, mel_buf);
  float melf_ms = (micros() - t0) / 1000.0f;
  float conf, inf_ms;
  int cls = run_inference(mel_buf, &conf, &inf_ms);
  Serial.printf("[mic] pred=%-14s conf=%.2f  peak=%ld rms=%d  mel=%.1fms infer=%.1fms total=%.1fms\n",
                cls >= 0 ? CLASSES[cls] : "ERR", conf, (long)peak, rms, melf_ms, inf_ms, melf_ms + inf_ms);
  emit_result(cls, conf, melf_ms + inf_ms, "mic");
  // wifi_bridge_push(CLASSES[cls], cls, conf, melf_ms + inf_ms);
#endif
}
