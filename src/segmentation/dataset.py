from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_mask_pairs(split_root: str | Path) -> List[Tuple[Path, Path]]:
    root = Path(split_root)
    images_dir = root / "images"
    masks_dir = root / "masks"

    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


class SegmentationDataset(Dataset):
    def __init__(
        self,
        split_root: str | Path,
        image_size: int,
        train: bool = False,
        horizontal_flip_prob: float = 0.0,
        color_jitter: bool = False,
    ) -> None:
        self.pairs = list_image_mask_pairs(split_root)
        self.image_size = image_size
        self.train = train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter

        if not self.pairs:
            raise ValueError(f"No image/mask pairs found in {split_root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self._resize(image, mask)

        if self.train:
            image, mask = self._augment(image, mask)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image_tensor, mask_tensor

    def _resize(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        size = (self.image_size, self.image_size)
        image = image.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self.color_jitter:
            image = self._apply_color_jitter(image)

        if random.random() < 0.05:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.8))

        return image, mask

    @staticmethod
    def _apply_color_jitter(image: Image.Image) -> Image.Image:
        for enhancer_factory, low, high in (
            (ImageEnhance.Brightness, 0.85, 1.15),
            (ImageEnhance.Contrast, 0.85, 1.15),
            (ImageEnhance.Color, 0.85, 1.15),
        ):
            enhancer = enhancer_factory(image)
            image = enhancer.enhance(random.uniform(low, high))
        return image

