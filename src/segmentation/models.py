from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name != "deeplabv3_resnet50":
        raise ValueError(f"Unsupported model: {name}")

    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, aux_loss=True)
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        aux_in_channels = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
    return model


def load_model_state(model: nn.Module, model_state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    current_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in model_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    load_result = model.load_state_dict(compatible_state, strict=False)
    missing_keys = list(load_result.missing_keys)
    return missing_keys, skipped_keys
