from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class SegmentationMetrics:
    mean_iou: float
    pixel_accuracy: float
    mean_dice: float
    per_class_iou: List[float]
    per_class_dice: List[float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "mean_iou": self.mean_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "mean_dice": self.mean_dice,
            "per_class_iou": self.per_class_iou,
            "per_class_dice": self.per_class_dice,
        }


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = predictions.detach().cpu().flatten()
        targets = targets.detach().cpu().flatten()
        valid = targets != self.ignore_index
        valid &= targets >= 0
        valid &= targets < self.num_classes

        targets = targets[valid]
        predictions = predictions[valid]
        predictions = predictions.clamp(min=0, max=self.num_classes - 1)

        indices = self.num_classes * targets + predictions
        bins = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += bins.reshape(self.num_classes, self.num_classes)

    def compute(self) -> SegmentationMetrics:
        matrix = self.matrix.float()
        true_positive = torch.diag(matrix)
        false_positive = matrix.sum(dim=0) - true_positive
        false_negative = matrix.sum(dim=1) - true_positive

        union = true_positive + false_positive + false_negative
        support = matrix.sum(dim=1)
        iou = torch.where(union > 0, true_positive / union.clamp_min(1), torch.nan)
        dice_denominator = 2 * true_positive + false_positive + false_negative
        dice = torch.where(
            dice_denominator > 0,
            2 * true_positive / dice_denominator.clamp_min(1),
            torch.nan,
        )

        total = matrix.sum().clamp_min(1)
        pixel_accuracy = true_positive.sum() / total

        valid_iou = iou[~torch.isnan(iou)]
        valid_dice = dice[~torch.isnan(dice)]

        return SegmentationMetrics(
            mean_iou=float(valid_iou.mean().item()) if valid_iou.numel() else 0.0,
            pixel_accuracy=float(pixel_accuracy.item()),
            mean_dice=float(valid_dice.mean().item()) if valid_dice.numel() else 0.0,
            per_class_iou=[float(value) if not torch.isnan(value) else 0.0 for value in iou],
            per_class_dice=[float(value) if not torch.isnan(value) else 0.0 for value in dice],
        )

