from __future__ import annotations

import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name != "deeplabv3_resnet50":
        raise ValueError(f"Unsupported model: {name}")

    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights)
    in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    return model

