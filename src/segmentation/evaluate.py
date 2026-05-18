from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_classes, load_yaml
from .dataset import SegmentationDataset
from .metrics import ConfusionMatrix
from .models import build_model, load_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a semantic segmentation checkpoint.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="outputs/reports/evaluation.json")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    classes, ignore_index = load_classes(config["data"]["classes"])
    ignore_index = int(config["data"].get("ignore_index", ignore_index))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegmentationDataset(
        split_root=Path(config["data"]["root"]) / args.split,
        image_size=int(config["data"]["image_size"]),
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config["model"]["name"], len(classes), pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    missing_keys, skipped_keys = load_model_state(model, checkpoint["model_state"])
    if skipped_keys:
        print(f"Skipped incompatible checkpoint keys: {skipped_keys}")
    if missing_keys:
        print(f"Missing model keys initialized from defaults: {missing_keys}")
    model.eval()

    matrix = ConfusionMatrix(num_classes=len(classes), ignore_index=ignore_index)
    for images, masks in tqdm(loader, desc=f"evaluate:{args.split}"):
        images = images.to(device)
        outputs = model(images)["out"]
        predictions = outputs.argmax(dim=1)
        matrix.update(predictions, masks)

    metrics = matrix.compute().as_dict()
    metrics["split"] = args.split
    metrics["classes"] = [item.name for item in classes]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
