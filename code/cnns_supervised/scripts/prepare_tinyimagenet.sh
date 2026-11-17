#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TINYIMAGENET_ROOT:-}" ]]; then
  echo "ERROR: set TINYIMAGENET_ROOT, e.g. export TINYIMAGENET_ROOT=/path/to/tinyimagenet" >&2
  exit 2
fi
if [[ "$TINYIMAGENET_ROOT" == /ruta* ]]; then
  echo "ERROR: TINYIMAGENET_ROOT points to placeholder path: $TINYIMAGENET_ROOT" >&2
  exit 2
fi

ROOT="$TINYIMAGENET_ROOT"
DOWNLOAD_DIR="$ROOT/_download"
ZIP_PATH="$DOWNLOAD_DIR/tiny-imagenet-200.zip"
URL="https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DOWNLOAD_DIR"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Downloading TinyImageNet to $ZIP_PATH"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$ZIP_PATH" "$URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$URL" -o "$ZIP_PATH"
  else
    echo "ERROR: need wget or curl to download TinyImageNet." >&2
    exit 3
  fi
else
  echo "Using existing zip: $ZIP_PATH"
fi

if [[ ! -d "$DOWNLOAD_DIR/tiny-imagenet-200" ]]; then
  echo "Extracting TinyImageNet"
  unzip -q "$ZIP_PATH" -d "$DOWNLOAD_DIR"
fi

python - "$DOWNLOAD_DIR/tiny-imagenet-200" "$ROOT" <<'PY'
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not src.exists():
    raise SystemExit(f"missing extracted source: {src}")

train_dst = dst / "train"
val_dst = dst / "val"
train_dst.mkdir(parents=True, exist_ok=True)
val_dst.mkdir(parents=True, exist_ok=True)

for class_dir in sorted((src / "train").iterdir()):
    images = class_dir / "images"
    if not images.is_dir():
        continue
    out = train_dst / class_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for image in images.iterdir():
        if image.is_file():
            target = out / image.name
            if not target.exists():
                shutil.copy2(image, target)

ann = src / "val" / "val_annotations.txt"
images_dir = src / "val" / "images"
if not ann.exists():
    raise SystemExit(f"missing validation annotations: {ann}")
for line in ann.read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) < 2:
        continue
    filename, wnid = parts[0], parts[1]
    out = val_dst / wnid
    out.mkdir(parents=True, exist_ok=True)
    source = images_dir / filename
    target = out / filename
    if source.exists() and not target.exists():
        shutil.copy2(source, target)
PY

python "$SCRIPT_DIR/check_imagefolder_dataset.py" --root "$ROOT"
