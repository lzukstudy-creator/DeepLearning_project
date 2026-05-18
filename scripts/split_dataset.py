from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw image/mask pairs into train/val/test folders.")
    parser.add_argument("--images", default="data/raw/images")
    parser.add_argument("--masks", default="data/raw/masks")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            print(f"missing mask: {image_path.name}")
    return pairs


def copy_pairs(pairs: List[Tuple[Path, Path]], output_root: Path, split: str) -> None:
    images_out = output_root / split / "images"
    masks_out = output_root / split / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in pairs:
        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(mask_path, masks_out / mask_path.name)


def main() -> None:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    pairs = find_pairs(Path(args.images), Path(args.masks))
    if not pairs:
        raise ValueError("No valid image/mask pairs found.")

    random.Random(args.seed).shuffle(pairs)
    total = len(pairs)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    output_root = Path(args.output)
    for split, split_pairs in splits.items():
        copy_pairs(split_pairs, output_root, split)
        print(f"{split}: {len(split_pairs)} pairs")


if __name__ == "__main__":
    main()

