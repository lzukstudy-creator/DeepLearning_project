from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class ClassInfo:
    id: int
    name: str
    color: Tuple[int, int, int]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_classes(path: str | Path) -> tuple[List[ClassInfo], int]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    classes = [
        ClassInfo(
            id=int(item["id"]),
            name=str(item["name"]),
            color=tuple(int(channel) for channel in item["color"]),
        )
        for item in payload["classes"]
    ]
    classes.sort(key=lambda item: item.id)
    ignore_index = int(payload.get("ignore_index", 255))
    return classes, ignore_index

