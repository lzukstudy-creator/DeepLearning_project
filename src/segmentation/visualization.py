from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image


def colorize_mask(mask: np.ndarray, colors: Sequence[Tuple[int, int, int]]) -> Image.Image:
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        color_mask[mask == class_id] = color
    return Image.fromarray(color_mask)


def save_colorized_mask(mask: np.ndarray, colors: Sequence[Tuple[int, int, int]], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colorize_mask(mask, colors).save(output_path)

