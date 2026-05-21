from __future__ import annotations

import argparse
import random
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Tuple

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
    parser = argparse.ArgumentParser(description="Prepare Oxford-IIIT Pet detection data in YOLO format.")
    parser.add_argument("--download-dir", default="data/downloads/oxford_pet")
    parser.add_argument("--output-root", default="data/detection")
    parser.add_argument("--max-per-class", type=int, default=300)
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
    return 0 if breed_from_stem(stem) in CAT_BREEDS else 1


def parse_box(xml_path: Path) -> tuple[int, int, int, int, int, int]:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    box = root.find("object/bndbox")
    if box is None or width <= 0 or height <= 0:
        raise ValueError(f"Invalid annotation: {xml_path}")

    xmin = int(float(box.findtext("xmin", "0")))
    ymin = int(float(box.findtext("ymin", "0")))
    xmax = int(float(box.findtext("xmax", "0")))
    ymax = int(float(box.findtext("ymax", "0")))
    return width, height, xmin, ymin, xmax, ymax


def yolo_line(class_id: int, width: int, height: int, xmin: int, ymin: int, xmax: int, ymax: int) -> str:
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def collect_items(download_dir: Path) -> List[Tuple[Path, Path, int]]:
    images_dir = download_dir / "images"
    xmls_dir = download_dir / "annotations" / "xmls"
    items: List[Tuple[Path, Path, int]] = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        xml_path = xmls_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue
        items.append((image_path, xml_path, class_id_from_stem(image_path.stem)))
    return items


def balanced_subset(items: Iterable[Tuple[Path, Path, int]], max_per_class: int, seed: int) -> List[Tuple[Path, Path, int]]:
    rng = random.Random(seed)
    cats = [item for item in items if item[2] == 0]
    dogs = [item for item in items if item[2] == 1]
    rng.shuffle(cats)
    rng.shuffle(dogs)
    selected = cats[:max_per_class] + dogs[:max_per_class]
    rng.shuffle(selected)
    return selected


def split_items(
    items: List[Tuple[Path, Path, int]],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, List[Tuple[Path, Path, int]]]:
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def write_split(split: str, items: List[Tuple[Path, Path, int]], output_root: Path) -> None:
    images_out = output_root / "images" / split
    labels_out = output_root / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_path, xml_path, class_id in items:
        width, height, xmin, ymin, xmax, ymax = parse_box(xml_path)
        shutil.copy2(image_path, images_out / image_path.name)
        label_path = labels_out / f"{image_path.stem}.txt"
        label_path.write_text(
            yolo_line(class_id, width, height, xmin, ymin, xmax, ymax) + "\n",
            encoding="utf-8",
        )

    print(f"{split}: {len(items)} samples")


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

    items = collect_items(download_dir)
    selected = balanced_subset(items, args.max_per_class, args.seed)
    splits = split_items(selected, args.train_ratio, args.val_ratio)
    output_root = Path(args.output_root)
    for split, split_items_for_name in splits.items():
        write_split(split, split_items_for_name, output_root)


if __name__ == "__main__":
    main()
