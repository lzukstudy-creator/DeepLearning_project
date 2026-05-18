from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate semantic segmentation dataset folders.")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--classes", default="configs/classes.json")
    return parser.parse_args()


def load_valid_ids(path: Path) -> tuple[Set[int], int]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    ids = {int(item["id"]) for item in payload["classes"]}
    ignore_index = int(payload.get("ignore_index", 255))
    return ids, ignore_index


def validate_split(split_root: Path, valid_ids: Set[int], ignore_index: int) -> Dict[str, int]:
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    counts = {
        "images": 0,
        "valid_pairs": 0,
        "missing_masks": 0,
        "size_mismatches": 0,
        "invalid_class_masks": 0,
    }

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        counts["images"] += 1
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            counts["missing_masks"] += 1
            print(f"[{split_root.name}] missing mask: {image_path.name}")
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                counts["size_mismatches"] += 1
                print(f"[{split_root.name}] size mismatch: {image_path.name} image={image.size} mask={mask.size}")

            mask_values = set(int(value) for value in np.unique(np.array(mask.convert("L"))))
            invalid_values = mask_values - valid_ids - {ignore_index}
            if invalid_values:
                counts["invalid_class_masks"] += 1
                print(f"[{split_root.name}] invalid class ids in {mask_path.name}: {sorted(invalid_values)}")

        counts["valid_pairs"] += 1

    return counts


def main() -> None:
    args = parse_args()
    valid_ids, ignore_index = load_valid_ids(Path(args.classes))
    data_root = Path(args.data_root)

    total_errors = 0
    for split in ("train", "val", "test"):
        counts = validate_split(data_root / split, valid_ids, ignore_index)
        total_errors += counts["missing_masks"] + counts["size_mismatches"] + counts["invalid_class_masks"]
        print(f"{split}: {counts}")

    if total_errors:
        raise SystemExit(f"dataset validation failed with {total_errors} issue(s)")
    print("dataset validation passed")


if __name__ == "__main__":
    main()

