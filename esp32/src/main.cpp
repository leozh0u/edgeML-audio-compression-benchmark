// ESC-50 edge classifier — ESP32-S3 firmware scaffold.
//
// Pipeline:  I2S mic (INMP441) -> 5s ring buffer -> log-mel spectrogram (esp-dsp
//            FFT + mel filterbank) -> TFLite Micro INT8 inference -> top-1 class
//            -> push {label, confidence, latency} to the dashboard over WebSocket.
//
// STATUS: structurally complete and flash-ready. mel_frontend.cc now does the
// real FFT/mel math (off-board validated against librosa, see
// scripts/validate_mel_frontend.py) instead of the old zero-fill stub, and
// the milestone-4 TFLite export hang is resolved (see DECISIONS.md §0),
// so a trustworthy int8 .tflite exists to embed via export_esp32.py.
// int8 quant params (input scale/zero-point) are read from the .tflite at
// runtime via input->params (below), never hardcoded.
//
// Nothing here needs the network or mic to *compile*; those are guarded so you
// can flash and confirm the model loads + runs on a static test input first.
// Wiring a real I2S mic buffer into mel_frontend_compute() and measuring
// on-device latency/RAM are the actual remaining board-only steps.

#include <Arduino.h>
#include "config.h"
#include "mel_frontend.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esc50_model/model_data.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_KB * 1024];

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
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nESC-50 edge classifier booting...");

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("model schema %lu != supported %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
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

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, sizeof(tensor_arena));
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed — raise TENSOR_ARENA_KB");
    return;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.printf("model loaded. arena used: %u bytes. input: [%d,%d,%d,%d] type=%d\n",
                interpreter->arena_used_bytes(),
                input->dims->data[0], input->dims->data[1],
                input->dims->data[2], input->dims->data[3], input->type);

  mel_frontend_init();
  // wifi_bridge_init();   // enable once WIFI_* in config.h is set
}

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

static int run_inference(const float* mel, float* out_confidence) {
  fill_input_int8(mel);
  uint32_t t0 = micros();
  if (interpreter->Invoke() != kTfLiteOk) { Serial.println("Invoke failed"); return -1; }
  uint32_t dt = micros() - t0;

  const float oscale = output->params.scale;
  const int ozp = output->params.zero_point;
  int best = 0; float best_val = -1e9f;
  for (int c = 0; c < N_CLASSES; ++c) {
    float logit = (output->data.int8[c] - ozp) * oscale;
    if (logit > best_val) { best_val = logit; best = c; }
  }
  *out_confidence = best_val;           // pre-softmax; dashboard can softmax if needed
  Serial.printf("pred=%-16s conf=%.2f  latency=%.1fms\n",
                CLASSES[best], best_val, dt / 1000.0f);
  return best;
}

void loop() {
  // TODO(board): capture 5s from the I2S mic into a ring buffer, then:
  //   mel_frontend_compute(pcm_buffer, mel_buf);   // fills mel_buf[N_MELS*N_FRAMES]
  // For bring-up without a mic, feed a constant frame to prove the graph runs:
  for (int i = 0; i < N_MELS * N_FRAMES; ++i) mel_buf[i] = 0.0f;

  float conf;
  int cls = run_inference(mel_buf, &conf);
  (void)cls;
  // wifi_bridge_push(CLASSES[cls], cls, conf, latency_ms);
  delay(1500);
}
