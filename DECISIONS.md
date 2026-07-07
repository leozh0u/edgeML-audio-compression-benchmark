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
  2. A trustworthy fully-INT8 .tflite for the QAT students still isn't produced (the TF SavedModel hang from milestone 4 is unresolved). Existing mid_student tflites are from the flaky TF path and their sizes look wrong (labeled "int8" but 251 KB). Regenerate in a clean env or Colab before trusting on-device numbers.

