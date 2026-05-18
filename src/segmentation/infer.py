from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .config import load_classes, load_yaml
from .models import build_model, load_model_state
from .visualization import draw_detection_box, save_colorized_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference on one image.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/predictions/result.png")
    parser.add_argument("--raw-output", default=None, help="Optional path for class-id grayscale mask.")
    parser.add_argument("--boxed-output", default=None, help="Optional path for image with label and red box.")
    return parser.parse_args()


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    classes, _ = load_classes(config["data"]["classes"])
    colors = [item.color for item in classes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config["model"]["name"], len(classes), pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    missing_keys, skipped_keys = load_model_state(model, checkpoint["model_state"])
    if skipped_keys:
        print(f"Skipped incompatible checkpoint keys: {skipped_keys}")
    if missing_keys:
        print(f"Missing model keys initialized from defaults: {missing_keys}")
    model.eval()

    original = Image.open(args.image).convert("RGB")
    input_tensor = preprocess(original, int(config["data"]["image_size"])).to(device)
    prediction = model(input_tensor)["out"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    prediction_image = Image.fromarray(prediction, mode="L")
    prediction_image = prediction_image.resize(original.size, Image.NEAREST)
    prediction = np.array(prediction_image, dtype=np.uint8)

    save_colorized_mask(prediction, colors, args.output)

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction, mode="L").save(raw_path)

    if args.boxed_output:
        draw_detection_box(
            image=original,
            mask=prediction,
            class_names=[item.name for item in classes],
            output=args.boxed_output,
        )


if __name__ == "__main__":
    main()
