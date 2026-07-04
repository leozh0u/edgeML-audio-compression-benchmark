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

