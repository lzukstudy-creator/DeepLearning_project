from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image

IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

CAT_BREEDS = {
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a cat/dog semantic segmentation subset.")
    parser.add_argument("--download-dir", default="data/downloads/oxford_pet")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"exists: {output}")
        return
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, output)


def extract_archive(archive: Path, marker: Path) -> None:
    if marker.exists():
        print(f"already extracted: {archive.name}")
        return
    print(f"extracting: {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(archive.parent)
    marker.touch()


def breed_from_stem(stem: str) -> str:
    return "_".join(stem.split("_")[:-1])


def class_id_from_stem(stem: str) -> int:
    return 1 if breed_from_stem(stem) in CAT_BREEDS else 2


def collect_pairs(download_dir: Path) -> List[Tuple[Path, Path, int]]:
    images_dir = download_dir / "images"
    trimaps_dir = download_dir / "annotations" / "trimaps"
    pairs = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        trimap_path = trimaps_dir / f"{image_path.stem}.png"
        if not trimap_path.exists():
            continue
        pairs.append((image_path, trimap_path, class_id_from_stem(image_path.stem)))
    return pairs


def balanced_subset(pairs: Iterable[Tuple[Path, Path, int]], max_per_class: int, seed: int) -> List[Tuple[Path, Path, int]]:
    rng = random.Random(seed)
    cats = [pair for pair in pairs if pair[2] == 1]
    dogs = [pair for pair in pairs if pair[2] == 2]
    rng.shuffle(cats)
    rng.shuffle(dogs)
    selected = cats[:max_per_class] + dogs[:max_per_class]
    rng.shuffle(selected)
    return selected


def split_pairs(
    pairs: List[Tuple[Path, Path, int]],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, List[Tuple[Path, Path, int]]]:
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }


def trimap_to_class_mask(trimap_path: Path, class_id: int) -> Image.Image:
    # Oxford trimap: 1=foreground, 2=background, 3=boundary/unknown.
    trimap = Image.open(trimap_path).convert("L")
    return trimap.point(lambda value: class_id if value == 1 else 255 if value == 3 else 0)


def write_split(split: str, pairs: List[Tuple[Path, Path, int]], output_root: Path) -> None:
    images_out = output_root / split / "images"
    masks_out = output_root / split / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    for image_path, trimap_path, class_id in pairs:
        shutil.copy2(image_path, images_out / image_path.name)
        mask = trimap_to_class_mask(trimap_path, class_id)
        mask.save(masks_out / f"{image_path.stem}.png")

    print(f"{split}: {len(pairs)} samples")


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    download_dir = Path(args.download_dir)
    images_archive = download_dir / "images.tar.gz"
    annotations_archive = download_dir / "annotations.tar.gz"

    download_file(IMAGES_URL, images_archive)
    download_file(ANNOTATIONS_URL, annotations_archive)
    extract_archive(images_archive, download_dir / ".images_extracted")
    extract_archive(annotations_archive, download_dir / ".annotations_extracted")

    pairs = collect_pairs(download_dir)
    selected = balanced_subset(pairs, args.max_per_class, args.seed)
    splits = split_pairs(selected, args.train_ratio, args.val_ratio)

    output_root = Path(args.output_root)
    for split, split_pairs_for_name in splits.items():
        write_split(split, split_pairs_for_name, output_root)


if __name__ == "__main__":
    main()
