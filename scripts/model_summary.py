from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.config import load_classes, load_yaml
from src.segmentation.models import build_model, load_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and visualize model parameter counts.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-json", default="outputs/reports/model_summary.json")
    parser.add_argument("--output-image", default="outputs/reports/model_parameters.png")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> tuple[int, int, Dict[str, int]]:
    total = 0
    trainable = 0
    by_top_module: Dict[str, int] = {}

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count

        top_module = name.split(".", 1)[0]
        by_top_module[top_module] = by_top_module.get(top_module, 0) + count

    return total, trainable, by_top_module


def save_chart(by_top_module: Dict[str, int], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(by_top_module.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values_millions = [item[1] / 1_000_000 for item in sorted_items]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values_millions, color=["#d62728", "#1f77b4", "#2ca02c"][: len(labels)])
    plt.ylabel("Parameters (millions)")
    plt.title("Model Parameters by Top-Level Module")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values_millions):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}M",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    classes, _ = load_classes(config["data"]["classes"])
    model = build_model(config["model"]["name"], len(classes), pretrained=False)

    checkpoint_info = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        missing_keys, skipped_keys = load_model_state(model, checkpoint["model_state"])
        checkpoint_info = {
            "path": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
            "missing_keys": missing_keys,
            "skipped_keys": skipped_keys,
        }

    total, trainable, by_top_module = count_parameters(model)
    summary = {
        "model": config["model"]["name"],
        "num_classes": len(classes),
        "classes": [item.name for item in classes],
        "total_parameters": total,
        "total_parameters_millions": round(total / 1_000_000, 4),
        "trainable_parameters": trainable,
        "trainable_parameters_millions": round(trainable / 1_000_000, 4),
        "parameters_by_top_module": by_top_module,
        "checkpoint": checkpoint_info,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_chart(by_top_module, args.output_image)
    print(json.dumps(summary, indent=2))
    print(f"Saved chart: {args.output_image}")


if __name__ == "__main__":
    main()
