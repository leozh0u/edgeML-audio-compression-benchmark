# Project Decisions Log

## Dataset
- Chose ESC-50 over UrbanSound8K/cough datasets: clean labels, fast iteration, benchmarked baselines available.

## Hardware
- ESP32-S3 (N16R8) chosen over STM32F401RE as primary deploy target: wifi built in, NN-friendlier clock (240MHz vs 84MHz), more RAM/flash headroom (16MB/8MB PSRAM vs 96KB SRAM).
- F401RE considered as secondary cross-platform benchmark target (not yet started).

## Milestone 1: Baseline teacher
- Architecture: CNN, 32->64->128->128 channels, mel-spectrogram input (64 mel bins, 5s clips, 22050 Hz).
- Split: ESC-50 standard protocol, folds 1-4 train / fold 5 val.
- Augmentation: SpecAugment (freq + time masking).
- Optimizer: Adam, lr 1e-3, cosine schedule, 30 epochs.
- Result: 71.3% val_acc. Train/val gap ~17pts (88.8% vs 71.3%) — mild overfit.

## Milestone 2: Improved teacher
- Architecture: wider CNN, 32->64->128->256->256 channels, +dropout(0.3).
- Augmentation: SpecAugment + mixup (alpha=0.3).
- Optimizer: AdamW, weight_decay 1e-4, 50 epochs.
- Result: 77.5% val_acc. Train/val gap tightened to ~7pts (84.7% vs 77.5%) — mixup reduced overfit.
- Decision: stopped tuning here. In line with published from-scratch ESC-50 CNN baselines (75-85% range). Diminishing returns vs. time cost of further tuning; moved to compression stage.
- Checkpoint: teacher_v2_best.pt selected as final teacher for distillation.

## Bugs encountered
- Two path mismatches: scripts assumed a directory layout that didn't exist on the training machine. Fix: always mkdir/verify on actual machine before running.
- Lost first training run: user multitasked (git/ssh) in the same terminal tab running training, killed the process.
- ModuleNotFoundError on second attempt: new terminal tab, venv not activated.
- GitHub HTTPS password auth failed (deprecated). Switched to SSH key auth.

## Repo
- github.com/leozh0u/edgeML-audio-compression-benchmark
- .gitignore excludes: venv, wandb logs, checkpoints, ESC-50 dataset, __pycache__.

## Milestone 3: Distillation
- TinyStudent (8->16->32->32 channels): 16,962 params, 58.5x compression vs teacher, 56.5% val_acc.
- MidStudent (16->32->64->64 channels): 63,826 params, 15.5x compression vs teacher, 70.5% val_acc.
- Pareto curve confirmed: teacher 992,242 params/77.5% acc, mid 15.5x/70.5%, tiny 58.5x/56.5%.

## Milestone 4: TFLite/LiteRT export attempt - abandoned
- Attempted PyTorch -> ONNX -> onnx-tf -> TFLite. onnx-tf is unmaintained, incompatible with current onnx (missing `mapping` module). Abandoned.
- Switched to onnx2tf (actively maintained). Hit a long dependency chain (tf_keras, psutil, ai_edge_litert, etc.) and, more critically, a reproducible hang: TensorFlow's SavedModel "Restoring SavedModel bundle" step froze indefinitely (0% CPU, unkillable via Ctrl+C, required force-quitting Terminal) on this machine, across multiple different code paths that all eventually called TF's SavedModel loader (onnx2tf's -osd flag, tf.lite.TFLiteConverter.from_saved_model, and onnx2tf's internal pipeline itself).
- Decision: abandoned the TFLite/onnx2tf path for now rather than continuing to debug an environment-specific TF SavedModel loader bug. Real LiteRT export deferred to when the ESP32-S3 board arrives (may retry in a fresh env, or use Google Colab if the hang persists locally).

## Milestone 4 alt: PTQ via ONNX Runtime static quantization
- Bypassed TensorFlow entirely: used onnxruntime.quantization.quantize_static (QDQ format, QInt8 weights + activations) directly on exported ONNX files. No TF/SavedModel involved, no hang.
- MidStudent: FP32 250.5 KB / 70.5% acc -> INT8 71.0 KB / 71.0% acc. 3.53x size reduction, accuracy essentially unchanged (+0.5pt, within noise).
- TinyStudent: FP32 19.3 KB / 56.5% acc -> INT8 37.6 KB / 52.75% acc. Quantization made this model WORSE on both axes (bigger file, lower accuracy).
  - Root cause: QDQ format inserts scale/zero-point overhead nodes around every op. At ~17K params, this per-op overhead outweighs the actual weight compression, so the file grows instead of shrinking.
  - This is a genuine finding, not a bug: quantization has a size floor below which it stops helping. Worth stating explicitly in the writeup as evidence of understanding the tradeoff, not glossed over.
  - Considered but did not yet try: QuantFormat.QOperator (less per-op overhead) as a potential fix for the tiny model specifically.
- Note: ONNX Runtime INT8 is a legitimate PTQ measurement but is not the final ESP32 deployment format (that's TFLite Micro / LiteRT). Treat these numbers as the PTQ benchmark point, separate from the eventual on-device deploy step.

## Repo recovery: 4 training scripts had been destroyed by a bad revert
- The distillation/QAT-relevant scripts (train_teacher_v2, distill_student, distill_student_mid, quantize_ptq) were committed in 81c3e2c under a wrong, copy-pasted message ("add idempotent confirm endpoint with replay detection"), then blind-reverted 30s later in b5a155d. The revert deleted the real work; only the confusing message survived to explain it.
- Recovered with `git checkout 81c3e2c -- <files>`. Lesson: never `git revert` on a bad-looking commit message without reading the diff.

## Environment drift: project moved, every script broke
- The repo was relocated from ~/edge-ml-esc50 (handoff layout) to ~/Projects/edgeml-audio/edge-ml-esc50, and ESC-50 to a sibling folder. Every script hardcoded os.path.expanduser("~/ESC-50") and "~/edge-ml-esc50/...", so all of them silently pointed at non-existent paths.
- Fix: added scripts/common.py with location-independent path resolution (derived from __file__, sibling-ESC-50 with ~/ fallback) and patched every legacy script's path block. common.py is now the single source of truth for paths, feature extraction, dataset, and the (frozen) model architectures.
- The second venv (tflite_env, py3.11) no longer exists on disk. Not needed for PTQ anymore: onnx + onnxruntime now have Python 3.14 wheels and were installed straight into esc50env. tflite_env is only needed if/when the TF->TFLite conversion path is revisited.

## Milestone 5: QAT (quantization-aware training) — the payoff
- FX-graph-mode QAT (prepare_qat_fx/convert_fx, qnnpack backend), fine-tuned with the SAME distillation loss as the original students. Ran on CPU (qnnpack int8 is CPU-only). Feature cache (precompute_features.py) makes CPU epochs ~5-7s.
- Gotcha: deepcopy of a prepared QAT module after a backward pass raises "Only Tensors created explicitly by the user support deepcopy." Fix: select the best epoch on the fake-quant model as an INT8 proxy, snapshot a detached state_dict, and convert to real INT8 exactly once at the end.
- Results (torch state_dict sizes; format differs from ONNX numbers above):
  - MidStudent : FP32 70.5% / 260.6 KB -> QAT-INT8 70.75% / 70.9 KB (3.68x smaller, +0.25pt).
  - TinyStudent: FP32 56.5% /  76.7 KB -> QAT-INT8 56.5%  / 25.0 KB (3.06x smaller,  0.0pt).
- THE headline finding: QAT rescued TinyStudent. Under PTQ, tiny got WORSE on both axes (56.5%->52.75%, grew to 37.6 KB). Under QAT it holds accuracy AND shrinks 3.06x. So the PTQ "quantization floor" for tiny models is not fundamental — fake-quant fine-tuning breaks through it. On the Pareto plot, tiny-qat sits on the frontier and tiny-ptq is strictly dominated by it. This is the strongest single result in the project.

## Milestone 6: Magnitude pruning + fine-tune
- Global unstructured L1 pruning across all conv/linear weights, then distillation fine-tune (10 ep). Swept sparsity 30/50/70/90%.
- Sizes reported as zlib-compressed state_dict (unstructured pruning zeros weights but doesn't shrink dense tensors; compressed size is the honest flash proxy). Structured/channel pruning (real dense speedup) noted as future work.
- Accuracy-vs-sparsity curves show clear cliffs and over-parameterization:
  - MidStudent : 30%->70.3, 50%->70.8 (best!), 70%->70.0, 90%->55.2. Can drop 70% of weights with ~no loss.
  - TinyStudent: 30%->57.0, 50%->56.8, 70%->53.5, 90%->25.0. Cliff between 50% and 70%.
- Interpretation: mid is heavily over-parameterized for ESC-50; tiny is near its capacity floor, so it tolerates far less pruning. Consistent with tiny being the model that PTQ couldn't compress either.

## Milestone 7: Benchmark harness + Pareto frontier
- scripts/benchmark.py merges every stage (fp32 evaluated live; ptq/qat/prune from their JSON) into results/benchmark.json, computes the accuracy-vs-size Pareto frontier, and renders results/pareto.png.
- Latency is HOST-CPU single-sample (torch for fp32, onnxruntime for ptq int8) — a portable proxy, explicitly NOT device latency. Real ESP32-S3 latency is added once the board is flashed.
- Frontier (accuracy vs size): teacher-fp32 77.5%/3.9MB -> mid-qat 70.8%/71KB -> tiny-qat 56.5%/25KB -> tiny-prune90 25%/15KB. QAT owns the sub-100KB region.

## Milestone 8: Dashboard (React + WebSocket)
- dashboard/ = zero-build React SPA (CDN React, React.createElement) + a stdlib-HTTP + websockets backend (server.py). Serves the Pareto scatter, the full variant table, and a live-inference panel.
- Live predictions arrive over WebSocket. --simulate emits synthetic predictions now so the whole loop is demonstrable pre-hardware; the ESP32 bridge just calls server.broadcast(pred) later. One-way device->UI flow, but kept on WebSocket (not SSE) deliberately for the frontend story.

## Milestone 3 (deploy plumbing): ESP32-S3 firmware scaffold — written pre-board
- esp32/ = PlatformIO project (esp32-s3-devkitc-1, PSRAM enabled): TFLite Micro inference, INMP441 I2S mic capture, WebSocket push to the dashboard, all params centralized in config.h to match training exactly.
- scripts/export_esp32.py converts a .tflite into a C byte array (model_data.cc/.h). Validated against the existing mid_student tflite.
- Two honest gaps that need the board / a clean int8 tflite:
  1. mel_frontend.cc — FFT/mel kernels are stubbed (control flow + normalization done); must be A/B-validated against wav_to_mel on-device or accuracy silently dies.
  2. A trustworthy fully-INT8 .tflite for the QAT students still isn't produced (the TF SavedModel hang from milestone 4 is unresolved). Existing mid_student tflites are from the flaky TF path and their sizes look wrong (labeled "int8" but 251 KB). Regenerate in a clean env or Colab before trusting on-device numbers. [RESOLVED below.]

## Milestone 4 RESOLVED: clean INT8 TFLite via native Keras rebuild
- Root cause of the original 5-hour hang was always the same: every route (onnx2tf -osd, tf.lite from_saved_model, onnx2tf internals) went through TensorFlow's *SavedModel-restore-from-disk* step, which froze indefinitely on this machine (0% CPU, unkillable).
- Fix (two-phase, avoids ONNX and SavedModel entirely):
  1. dump_weights.py (esc50env/torch): export each student's conv/BN/linear weights + K reference FP32 logits to a plain .npz. Keeps torch out of the TF env.
  2. convert_tflite_clean.py (tflite_env, TF 2.21 / py3.11): rebuild the student natively in Keras, port the weights (NCHW->NHWC: conv transpose (2,3,1,0), linear .T, BN eps forced to torch's 1e-5), then tf.lite.TFLiteConverter.from_keras_model with a representative dataset for full INT8. from_keras_model traces an in-memory concrete function — no SavedModel bundle is ever written or restored, so the hang is structurally impossible.
- Recreated tflite_env at ../tflite_env (py3.11, tensorflow 2.21). TF imports and converts in seconds; no hang. The "environment-specific TF SavedModel bug" was never in TF-the-library, it was that one on-disk-restore code path.
- Two guardrails make the output trustworthy (not just produced):
  1. FP32 parity BEFORE quantizing: Keras logits vs torch reference logits -> max|Δ| ~8e-6, 100% argmax agreement. Proves the port is faithful.
  2. INT8 validation on full fold-5 via tf.lite.Interpreter.
- Results (verified, full-INT8, int8 in/out — the actual deployable format):
  - mid_student_int8_clean.tflite : 70.0% acc, 74.5 KB.
  - tiny_student_int8_clean.tflite: 54.0% acc, 26.7 KB.
- Note: tiny at 54.0% (tflite) vs 56.5% (QAT) — consistent with the whole project's finding that tiny is the model that resists compression. QAT stays the best tiny *checkpoint*; the tflite is what deploys.
- On-device op set verified with the interpreter: CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED (ReLU fused into conv; int8 input so no QUANTIZE op). ESP32 firmware resolver trimmed to exactly these 4. model_data.cc regenerated from the clean 74.5 KB mid tflite.
- Old flaky-path tflites under tflite_models/mid_student_out/ and mid_student_savedmodel/ are left in place but superseded; use *_int8_clean.tflite.

## Milestone 3 (deploy plumbing), part 2: mel front end implemented + validated off-board
- Closed the last real software gap flagged in the M3 entry above: `esp32/src/mel_frontend.cc` was a zero-fill stub with only framing/normalization control flow. Replaced it with the real DSP pipeline, reproducing `common.py::wav_to_mel` (librosa `melspectrogram` + `power_to_db(ref=np.max)` + per-clip z-norm) in C:
  1. Framing with librosa's `center=True`, `pad_mode='constant'` semantics: signal is (conceptually) zero-padded by `n_fft//2` on each side, frame `f` starts at `f*hop_length - n_fft//2` in original-sample coordinates. This is why N_FRAMES=216, not 212 — got this wrong on first pass by assuming no centering, which is the off-by-a-few-frames bug an on-device-only debug session would have burned hours on.
  2. Window: periodic Hann (`0.5 - 0.5*cos(2*pi*n/N)`, divide by N not N-1). Checked numerically against `librosa.filters.get_window('hann', 2048, fftbins=True)`: periodic form matches to 2e-16, the symmetric form (divide by N-1, what the original stub had) is off by ~1e-3 per sample — small per-sample but compounds through the FFT/mel/log chain.
  3. FFT: wrote a portable radix-2 Cooley-Tukey complex FFT from scratch (`esp32/src/fft.c`) instead of taking the esp-dsp dependency immediately. Verified against `numpy.fft.fft` on a multi-tone test signal (2048-pt, DC + 10Hz + 100Hz bins): max abs error 0.006 against a max magnitude of 2048 (float32 rounding). Rationale: pure C means the exact same file compiles and runs on host and on-device, so it's unit-testable now instead of only after flashing. esp-dsp's `dsps_fft2r_fc32` does the identical radix-2 math with hardware-accelerated butterflies — swapping it in later is a speed optimization, not a correctness fix; noted as a follow-up if on-device profiling shows FFT is the bottleneck.
  4. Mel filterbank: confirmed librosa's 64 triangular filters are contiguous over the rfft bins (checked: 9-102 nonzero bins per filter, all contiguous, 1990/65600 nonzero total). Added `scripts/gen_mel_filterbank.py`, which calls `librosa.filters.mel(sr, n_fft, n_mels)` once on the host and emits a sparse (start_bin, weights[]) table as `esp32/src/mel_filterbank_data.h` — 8.5KB instead of a 262KB dense 64x1025 matrix. Depending on real librosa output here (not a hand-derived mel-scale formula) removes an entire class of Slaney-vs-HTK-mel-scale bugs.
  5. `power_to_db`: replicated librosa's exact formula, including the part the original stub was missing — `ref=np.max` (subtract 10*log10(global max power) from every bin) and the `top_db=80` floor clip (`log_spec = max(log_spec, log_spec.max() - 80)`). The stub only had the final z-norm; without the db conversion and its two clip on the global max, the network sees a completely different input distribution than it trained on.
- Validation methodology (all off-board, no ESP32 needed): `esp32/host_test/harness.cc` is a ~40-line native (no ESP-IDF) driver that calls the exact same `mel_frontend_init`/`mel_frontend_compute` used on-device. `scripts/validate_mel_frontend.py` compiles it with `cc`/`c++`, generates 4 synthetic 5s test signals (multi-tone, white noise, a silence+transient burst, a chirp), runs each through both the C harness and `common.wav_to_mel` (via a real WAV round-trip so librosa.load sees the same int16 quantization as the C code), and diffs the two mel spectrograms.
- Result: max|Δ| ≤ 0.0009, mean|Δ| ≈ 0.0000, correlation ≥ 0.99999 on all 4 cases (all in the z-normalized output space). This is float32-rounding-level agreement, not "close enough" — the on-device math is provably the same computation as training, modulo float rounding.
- Honest limit of this validation: it's synthetic PCM through both paths, not a real microphone signal. It proves the DSP *math* is correct; it does not and cannot prove I2S/ADC capture behavior (clock drift, mic self-noise, gain staging) is correct — that genuinely needs the board and is now the only mel-front-end item left in "what needs the board".
- `esp32/platformio.ini`: dropped the `espressif/esp-dsp` lib_dep since it's unused now (portable FFT instead); left a comment for re-adding it if a later speed optimization needs it.

## Documentation: public README
- Added `README.md` as the public front door: headline compression numbers, the PTQ-vs-QAT finding, the full 17-variant results table, a pipeline diagram, and repro steps. Rationale: a portfolio repo with no README reads as unfinished no matter how good the code is; the README is the one artifact a reader actually opens first. Every number is sourced from `results/benchmark.json`, not restated from memory, so it can't drift from the harness output. This log (DECISIONS.md) stays the detailed engineering narrative behind those numbers.

## Milestone 8 verified (not just written): dashboard end-to-end check
- Ran `dashboard/server.py --simulate` and exercised every path rather than trusting that it still worked after the repo moves/churn: HTTP serves the SPA and `/api/benchmark` returns all 17 variants; the WebSocket completes its `hello` handshake (50 ESC-50 classes) and streams live simulated predictions (`{label, confidence, latency_ms, source}`). So the "React + live WebSocket inference" story is demonstrable today, before the board — `--simulate` stands in for the device feed, and the ESP32 bridge only has to call `broadcast()`.
- Open decision logged for the next session: the benchmark's latency axis only covers `fp32`/`ptq` (host-CPU proxy); `qat`/`prune`/`tflite_int8` are `null`. Either measure host-CPU tf.lite-interpreter latency for the two clean int8 tflites (meaningful and cheap) to thicken the axis, or reframe the frontier as accuracy/size with latency explicitly deferred to on-device. Host-CPU latency for qnnpack QAT int8 models was judged not meaningful enough to report. [RESOLVED below.]

## Latency axis: measured host-CPU latency for the deployable tflites
- Added `scripts/measure_tflite_latency.py`: times the two clean int8 tflites with `tf.lite.Interpreter` (single thread, batch 1, int8 in/out — the exact graph that ships to the ESP32), 10 warmup + 200 timed invokes, and writes `int8_latency_ms` into `results/tflite_results.json`; `benchmark.py` now carries it into the unified table.
- Results: mid 0.346 ms, tiny 0.241 ms host-CPU — sits plausibly between the onnxruntime ptq numbers (~0.15–0.18 ms) and torch fp32 (~1.0–2.2 ms). Same proxy caveat as the whole harness: comparable across variants on this machine, not a device number.
- Deliberately did NOT fabricate latency for `qat` (qnnpack int8 through the torch stack measures torch dispatch overhead more than model cost) or `prune` rows (unstructured zeros don't change dense compute time — reporting a "pruned latency" would imply a speedup that doesn't exist). Those stay null on purpose; that's the honest shape of unstructured pruning.
- Latency proxy caveat within the caveat: fp32 rows are re-measured on every `benchmark.py` run, while ptq/tflite latencies come from their stage JSONs (measured in separate sessions, machine load varies). Cross-variant comparisons are order-of-magnitude, not benchmark-grade; the device numbers are the real ones.


## Milestone 9: ESP32-S3 hardware bring-up — the estimates were wrong, and that's the finding
Board in hand (ESP32-S3-DevKitC-1, N16R8: 16 MB flash, 8 MB PSRAM, 240 MHz, rev v0.2). This is the first time the firmware was ever *compiled and run*, and it surfaced two real bugs plus the deployment tradeoff the whole project was building toward. Every number below is measured on the actual chip.

**Toolchain / bring-up friction (logged so it isn't re-suffered):**
- PlatformIO installed into a dedicated `../pioenv` (py3.11); board flashes over the DevKitC's **UART** port (`/dev/cu.usbserial-*`, FTDI), not the native-USB port — `Serial` maps to UART0 by default, so the native-USB port shows nothing.
- The framework download (~200 MB) stalled mid-transfer on the VPN and hung *pio* for 13 h on a dead socket. Fix: a watchdog wrapper that kills+retries `pio run` when `~/.platformio` stops growing for 120 s. Toolchain is cached now, so this is a one-time cost.

**Two scaffold bugs the pre-board firmware hid (it had never been compiled):**
1. `MicroInterpreter` was constructed with the modern 4-arg signature, but `tanakamasayuki/TensorFlowLite_ESP32` tracks an older TFLite Micro API that still requires an `ErrorReporter*`. Added a `MicroErrorReporter`. Lesson: an uncompiled "flash-ready" scaffold is a hypothesis, not a fact.
2. The 200 KB tensor arena was a static `.bss` array → linker error, `dram0_0_seg overflowed`. The ESP32-S3 has only ~320 KB internal DRAM and ~103 KB is already `.bss` (framework/wifi/mel buffers). Moved the arena to a runtime `heap_caps_malloc`, preferring internal SRAM and falling back to PSRAM.

**The deployment tradeoff, measured (this is the headline of the hardware phase):**
- **Mid model (74.5 KB tflite, 70% acc):** `arena_used_bytes` = **273 KB**. That does not fit internal SRAM (only ~282 KB free, i.e. 97% utilization — no room for wifi/mic buffers), so it can only live in PSRAM. Inference from PSRAM: **22,563 ms (22.5 s)**. PSRAM bandwidth, not compute, dominates when every intermediate feature map is off-chip.
- **Tiny model (26.7 KB tflite, 54% acc):** `arena_used_bytes` = **137 KB**, fits internal SRAM with ~84 KB heap to spare. Inference: **6,148 ms (6.1 s)** — 3.7× faster than mid-in-PSRAM purely from arena locality.
- Both models run the **full on-device pipeline** (C mel front end → int8 inference) correctly on real embedded ESC-50 audio: `dog` and `church_bells` both classify correctly on-device (2/2), matching the host tflite prediction through the exact C DSP. **This closes the #1 risk item** — the mel front end is right not just on synthetic PCM but on real 16-bit audio through the on-chip float FFT. (Physical I2S mic capture is still unwired; that's the only remaining board item.)

**Why 6.1 s and not the ~300 ms I estimated:** the estimate was simply wrong — it assumed optimized kernels. `tanakamasayuki/TensorFlowLite_ESP32` ships only **reference** int8 kernels (no SIMD); its `kernels/esp_nn/` directory is a README placeholder, not the real esp-nn implementation. The tiny model is ~240 M int8 MACs of conv; reference kernels at ~single-MAC-per-few-cycles put that in the multi-second range, and the identical per-clip latency (6148.2 ms every time, input-independent) confirms it's pure compute, not the mel front end. **The identified fix — not done, flagged as the next optimization — is Espressif's `esp-tflite-micro`, which bundles esp-nn (S3 SIMD kernels), expected 5–20× → sub-second.** That's a library swap with real integration risk, deliberately not attempted unsupervised over a flaky VPN.

**Deployment decision:** ship the **tiny** model on-device (fits fast internal SRAM, runs, real-time-ish only after esp-nn). The mid model stays the headline *accuracy/size* result; the *on-device* story is tiny + the measured mid-doesn't-fit-RAM reason. This is exactly the accuracy/latency/**memory** trilemma the Pareto framing promised — now with silicon numbers, not host proxies.

**Resume numbers corrected (these replace the ~300 ms / ~300 KB estimates everywhere):**
- On-device model: **tiny INT8, 26.7 KB, 54%** (not mid — mid doesn't fit fast RAM).
- On-device inference latency: **6.1 s** with reference kernels (→ sub-second projected with esp-nn).
- On-device RAM: **137 KB tensor arena (internal SRAM) + ~103 KB static ≈ 240 KB** of 320 KB.
- Flash: **816 KB** for the self-test build (of which ~431 KB is the 2 embedded demo clips; a mic-based production build is ~385 KB).

**Firmware now has 3 build-time modes** (`-DBRINGUP_MODE`): `1` model-check (zero input), `2` self-test (embedded real clips → full pipeline, the default), `0` mic (live I2S capture, written but needs the mic wired + gain calibration). Each prediction also emits a machine-readable `RESULT {json}` line.

**Live dashboard, now real (not simulated):** added `server.py --serial <port>`, a background thread that parses the board's `RESULT` lines off USB and calls `broadcast()`. Verified end-to-end: board → USB serial → WebSocket → client streamed 5 live device predictions. The "live WebSocket inference from the device" bullet is demonstrable on hardware over the flashing cable alone — no wifi needed.
