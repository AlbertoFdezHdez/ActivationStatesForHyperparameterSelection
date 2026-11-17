#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CODE_ROOT/../.." && pwd)"

export STANFORDCARS_ROOT="${STANFORDCARS_ROOT:-$HOME/OUI_journal/datasets/stanford_cars}"
DOWNLOAD_DIR="$STANFORDCARS_ROOT/_download"
REPORT_PATH="$STANFORDCARS_ROOT/dataset_report.json"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${STANFORDCARS_ROOT:-}" == /ruta* ]]; then
  echo "STANFORDCARS_ROOT points to placeholder path: $STANFORDCARS_ROOT" >&2
  exit 2
fi

mkdir -p "$DOWNLOAD_DIR"

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Missing Kaggle credentials" >&2
  echo "Expected credentials at ~/.kaggle/kaggle.json, or an already configured Kaggle CLI session." >&2
  exit 3
fi

chmod 600 "$HOME/.kaggle/kaggle.json" || true

if ! command -v kaggle >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install -U kaggle
fi

kaggle datasets download \
  -d rickyyyyyyy/torchvision-stanford-cars \
  -p "$DOWNLOAD_DIR" \
  --unzip

"$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from torchvision.datasets import StanfordCars

root = Path(os.environ["STANFORDCARS_ROOT"]).expanduser()

target = root / "stanford_cars"

def is_torchvision_stanfordcars_dir(path: Path) -> bool:
    return (
        (path / "devkit" / "cars_train_annos.mat").exists()
        and (path / "devkit" / "cars_meta.mat").exists()
        and (path / "cars_train").is_dir()
        and (path / "cars_test").is_dir()
        and (path / "cars_test_annos_withlabels.mat").exists()
    )

if not is_torchvision_stanfordcars_dir(target):
    candidates = []
    for base in [root, root / "_download"]:
        if base.exists():
            candidates.extend([p for p in base.rglob("*") if p.is_dir() and is_torchvision_stanfordcars_dir(p)])
    candidates = [p for p in candidates if p.resolve() != target.resolve()]
    if not candidates:
        raise SystemExit(f"Dataset not found after Kaggle download. Expected torchvision StanfordCars layout under {target}")
    source = candidates[0]
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not any(target.iterdir()):
            target.unlink() if target.is_symlink() else target.rmdir()
        else:
            raise SystemExit(f"Invalid non-empty target exists: {target}. Move it aside and rerun.")
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)

train = StanfordCars(root=str(root), split="train", download=False)
test = StanfordCars(root=str(root), split="test", download=False)

report = {
    "root": str(root),
    "source": "kaggle:rickyyyyyyy/torchvision-stanford-cars",
    "num_classes": len(train.classes),
    "train_images": len(train),
    "test_images": len(test),
    "split_names": ["train", "test"],
    "classes_match": train.classes == test.classes,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
if report["num_classes"] != 196 or len(train) < 8000 or len(test) < 7900:
    raise SystemExit(f"StanfordCars verification failed: {report}")

(root / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY

echo "StanfordCars prepared at $STANFORDCARS_ROOT"
