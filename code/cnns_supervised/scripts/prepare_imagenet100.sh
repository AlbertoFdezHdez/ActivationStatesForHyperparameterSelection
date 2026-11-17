#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${IMAGENET100_ROOT:-}" ]]; then
  echo "ERROR: set IMAGENET100_ROOT, e.g. export IMAGENET100_ROOT=/path/to/imagenet100" >&2
  exit 2
fi
if [[ "$IMAGENET100_ROOT" == /ruta* ]]; then
  echo "ERROR: IMAGENET100_ROOT points to placeholder path: $IMAGENET100_ROOT" >&2
  exit 2
fi

ROOT="$IMAGENET100_ROOT"
DOWNLOAD_DIR="$ROOT/_download"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DOWNLOAD_DIR"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
elif command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
else
  echo "ERROR: huggingface_hub CLI not found. Install with: pip install -U 'huggingface_hub[cli]'" >&2
  exit 3
fi

echo "Checking Hugging Face login state. Public download may still work if this fails."
set +e
if [[ "$HF_CLI" == "huggingface-cli" ]]; then
  huggingface-cli whoami >/dev/null 2>&1
else
  hf auth whoami >/dev/null 2>&1
fi
whoami_status=$?
set -e
if [[ "$whoami_status" -ne 0 && -n "${HF_TOKEN:-}" ]]; then
  echo "Logging into Hugging Face using HF_TOKEN from environment."
  if [[ "$HF_CLI" == "huggingface-cli" ]]; then
    huggingface-cli login --token "$HF_TOKEN" >/dev/null
  else
    hf auth login --token "$HF_TOKEN" >/dev/null
  fi
fi

echo "Downloading asafaa/imagenet100-cmc to $DOWNLOAD_DIR"
set +e
if [[ "$HF_CLI" == "huggingface-cli" ]]; then
  huggingface-cli download asafaa/imagenet100-cmc --repo-type dataset --local-dir "$DOWNLOAD_DIR"
else
  hf download asafaa/imagenet100-cmc --repo-type dataset --local-dir "$DOWNLOAD_DIR"
fi
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
  echo "ERROR: Hugging Face download failed. If this was 401/403, export HF_TOKEN with a read token and rerun." >&2
  exit "$status"
fi

python - "$DOWNLOAD_DIR" "$ROOT" <<'PY'
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

download = Path(sys.argv[1])
root = Path(sys.argv[2])

for archive in list(download.rglob("*.zip")):
    marker = archive.with_suffix(archive.suffix + ".extracted")
    if marker.exists():
        continue
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(download)
    marker.write_text("ok\n", encoding="utf-8")

for archive in list(download.rglob("*.tar")) + list(download.rglob("*.tar.gz")) + list(download.rglob("*.tgz")):
    marker = archive.with_suffix(archive.suffix + ".extracted")
    if marker.exists():
        continue
    with tarfile.open(archive) as tf:
        tf.extractall(download)
    marker.write_text("ok\n", encoding="utf-8")

def find_split(name):
    candidates = [p for p in download.rglob(name) if p.is_dir()]
    for path in candidates:
        class_dirs = [p for p in path.iterdir() if p.is_dir()]
        if len(class_dirs) >= 90:
            return path
    return None

train = find_split("train")
val = find_split("val") or find_split("validation")
if train is None or val is None:
    raise SystemExit(
        "Could not find extracted train/val ImageFolder splits. "
        "Inspect $IMAGENET100_ROOT/_download and adapt extraction if repo layout changed."
    )

for split_name, source in [("train", train), ("val", val)]:
    target = root / split_name
    target.mkdir(parents=True, exist_ok=True)
    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        out = target / class_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for image in class_dir.rglob("*"):
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dest = out / image.name
                if not dest.exists():
                    shutil.copy2(image, dest)

for meta in download.rglob("*"):
    if meta.name.lower() in {"labels.json", "selected_synsets.txt", "imagenet100.txt"}:
        dest = root / meta.name
        if not dest.exists():
            shutil.copy2(meta, dest)
PY

python "$SCRIPT_DIR/check_imagefolder_dataset.py" --root "$ROOT"
