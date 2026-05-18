from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_classes, load_yaml
from .dataset import SegmentationDataset
from .metrics import ConfusionMatrix
from .models import build_model, load_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume training.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loader(config: Dict, split: str, train: bool) -> DataLoader:
    data_root = Path(config["data"]["root"])
    training_config = config["training"]
    augmentation_config = config.get("augmentation", {})

    dataset = SegmentationDataset(
        split_root=data_root / split,
        image_size=int(config["data"]["image_size"]),
        train=train,
        horizontal_flip_prob=float(augmentation_config.get("horizontal_flip_prob", 0.0)) if train else 0.0,
        color_jitter=bool(augmentation_config.get("color_jitter", False)) if train else False,
    )

    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=train,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, device, ignore_index: int) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)["out"]
        loss = F.cross_entropy(outputs, masks, ignore_index=ignore_index)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, ignore_index: int) -> Dict[str, object]:
    model.eval()
    matrix = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    total_items = 0

    for images, masks in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]
        loss = F.cross_entropy(outputs, masks, ignore_index=ignore_index)
        predictions = outputs.argmax(dim=1)
        matrix.update(predictions, masks)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    metrics = matrix.compute().as_dict()
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: Dict[str, object],
    config: Dict,
    best_miou: float,
    history: list[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "best_miou": best_miou,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer, device) -> tuple[int, float, list[Dict[str, object]]]:
    checkpoint = torch.load(path, map_location=device)
    missing_keys, skipped_keys = load_model_state(model, checkpoint["model_state"])
    if skipped_keys:
        print(f"Skipped incompatible checkpoint keys while resuming: {skipped_keys}")
    if missing_keys:
        print(f"Missing model keys initialized from defaults while resuming: {missing_keys}")

    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    metrics = checkpoint.get("metrics", {})
    best_miou = float(checkpoint.get("best_miou", metrics.get("mean_iou", -1.0)))
    history = list(checkpoint.get("history", []))
    print(f"Resumed from {path}: next_epoch={start_epoch}, best_miou={best_miou:.4f}")
    return start_epoch, best_miou, history


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    classes, ignore_index = load_classes(config["data"]["classes"])
    ignore_index = int(config["data"].get("ignore_index", ignore_index))
    set_seed(int(config["training"].get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_loader(config, split="train", train=True)
    val_loader = create_loader(config, split="val", train=False)

    model = build_model(
        name=config["model"]["name"],
        num_classes=len(classes),
        pretrained=bool(config["model"].get("pretrained", True)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    best_miou = -1.0
    history = []
    start_epoch = 1

    if args.resume:
        start_epoch, best_miou, history = load_checkpoint(args.resume, model, optimizer, device)

    last_metrics: Dict[str, object] = {}

    try:
        for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, ignore_index)
            val_metrics = evaluate(model, val_loader, device, len(classes), ignore_index)
            val_metrics["train_loss"] = train_loss
            val_metrics["epoch"] = epoch
            history.append(val_metrics)
            last_metrics = val_metrics

            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_miou={val_metrics['mean_iou']:.4f}"
            )

            if float(val_metrics["mean_iou"]) > best_miou:
                best_miou = float(val_metrics["mean_iou"])
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, val_metrics, config, best_miou, history)

            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, val_metrics, config, best_miou, history)
    except KeyboardInterrupt:
        interrupted_epoch = max(start_epoch - 1, int(last_metrics.get("epoch", 0)))
        interrupted_metrics = last_metrics or {"interrupted": True, "epoch": interrupted_epoch}
        save_checkpoint(
            checkpoint_dir / "interrupted.pt",
            model,
            optimizer,
            interrupted_epoch,
            interrupted_metrics,
            config,
            best_miou,
            history,
        )
        print(f"\nTraining interrupted. Saved checkpoint to {checkpoint_dir / 'interrupted.pt'}")
        raise

    report_path = Path(config["training"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump({"history": history, "best_mean_iou": best_miou}, file, indent=2)


if __name__ == "__main__":
    main()
