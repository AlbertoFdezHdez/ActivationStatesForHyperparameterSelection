from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG"}


def count_images(class_dir: Path) -> int:
    return sum(1 for path in class_dir.rglob("*") if path.is_file() and path.suffix in IMAGE_EXTS)


def split_dir(root: Path, name: str) -> Path:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing split directory: {path}")
    return path


def class_counts(path: Path) -> dict[str, int]:
    classes = sorted(p for p in path.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in {path}")
    counts = {p.name: count_images(p) for p in classes}
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"Classes with zero images in {path}: {empty[:10]}")
    return counts


def build_report(root: Path) -> dict[str, object]:
    train = split_dir(root, "train")
    val_name = "val" if (root / "val").exists() else "validation"
    val = split_dir(root, val_name)
    train_counts = class_counts(train)
    val_counts = class_counts(val)
    train_classes = set(train_counts)
    val_classes = set(val_counts)
    if train_classes != val_classes:
        raise ValueError(
            "Train/validation class mismatch: "
            f"train_only={sorted(train_classes - val_classes)[:10]}, "
            f"val_only={sorted(val_classes - train_classes)[:10]}"
        )
    return {
        "root": str(root.resolve()),
        "num_classes": len(train_counts),
        "train_images": sum(train_counts.values()),
        "val_images": sum(val_counts.values()),
        "split_names": ["train", val_name],
        "class_ids": sorted(train_counts),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an ImageFolder dataset and write dataset_report.json.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    report = build_report(root)
    report_path = root / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=False))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
