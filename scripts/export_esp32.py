"""
Convert a .tflite model into a C source array for TFLite Micro on the ESP32-S3.

Produces esp32/esc50_model/model_data.{h,cc}: a `const unsigned char g_model[]`
byte array (16-wide rows) plus its length, which the firmware compiles directly
into flash. Equivalent to `xxd -i`, but self-contained and it also prints the
flash footprint so you can sanity-check it fits the target before flashing.

Usage:
  python scripts/export_esp32.py --tflite tflite_models/<model>.tflite

IMPORTANT (see DECISIONS.md): the local TF SavedModel->TFLite path hangs on this
machine, so a *trustworthy* fully-INT8 .tflite for the QAT students is not yet
produced. This script is the deploy-side plumbing and is validated against the
existing mid_student tflite; wire it to the real QAT-int8 .tflite once that is
generated (fresh env or Colab).
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "esp32" / "src" / "esc50_model"


def to_c_array(data: bytes, var="g_model") -> str:
    lines = [f"const unsigned char {var}[] = {{"]
    for i in range(0, len(data), 16):
        row = ", ".join(f"0x{b:02x}" for b in data[i:i + 16])
        lines.append(f"  {row},")
    lines.append("};")
    lines.append(f"const int {var}_len = {len(data)};")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True, help="path to the .tflite model")
    ap.add_argument("--var", default="g_model")
    args = ap.parse_args()

    tflite_path = Path(args.tflite)
    data = tflite_path.read_bytes()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "model_data.cc").write_text(
        '#include "model_data.h"\n\n'
        f'// Generated from {tflite_path.name} ({len(data)} bytes)\n'
        f"alignas(16) {to_c_array(data, args.var)}\n"
    )
    (OUT_DIR / "model_data.h").write_text(
        "#pragma once\n"
        f"extern const unsigned char {args.var}[];\n"
        f"extern const int {args.var}_len;\n"
    )

    kb = len(data) / 1024
    print(f"wrote {OUT_DIR/'model_data.cc'} and .h")
    print(f"model: {tflite_path.name}  flash footprint: {kb:.1f} KB")
    print(f"ESP32-S3 (N16R8) has 16 MB flash / 8 MB PSRAM -> fits with huge margin.")


if __name__ == "__main__":
    main()
