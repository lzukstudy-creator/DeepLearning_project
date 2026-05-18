from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def draw_detection_box(
    image: Image.Image,
    mask: np.ndarray,
    class_names: Sequence[str],
    output: str | Path,
    background_id: int = 0,
    min_pixels: int = 50,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_ids = [class_id for class_id in np.unique(mask) if class_id != background_id]
    if not candidate_ids:
        image.save(output_path)
        return

    class_id = int(
        max(
            candidate_ids,
            key=lambda value: int((mask == value).sum()),
        )
    )
    animal_pixels = mask == class_id
    if int(animal_pixels.sum()) < min_pixels:
        image.save(output_path)
        return

    y_positions, x_positions = np.where(animal_pixels)
    left = int(x_positions.min())
    top = int(y_positions.min())
    right = int(x_positions.max())
    bottom = int(y_positions.max())

    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

    line_width = max(3, min(result.size) // 150)
    draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=line_width)

    font = ImageFont.load_default()
    text_bbox = draw.textbbox((left, top), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_top = max(0, top - text_height - 8)
    draw.rectangle(
        (left, label_top, left + text_width + 10, label_top + text_height + 8),
        fill=(255, 0, 0),
    )
    draw.text((left + 5, label_top + 4), label, fill=(255, 255, 255), font=font)
    result.save(output_path)
