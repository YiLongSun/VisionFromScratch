"""Pascal VOC 共用常數與讀檔工具。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

VOC_CLASSES: Tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
# 類別編號從 1 開始，0 保留給背景（偵測與分割一致）
CLASS_TO_INDEX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}
CLASS_NAMES_WITH_BG: Tuple[str, ...] = ("background", *VOC_CLASSES)
NUM_CLASSES_WITH_BG = len(CLASS_NAMES_WITH_BG)  # 21
IGNORE_INDEX = 255  # 分割標註中物件邊界的「不計分」像素


def read_split_ids(root: str | Path, year: str, task_dir: str, split: str) -> List[str]:
    """讀取 ImageSets/<task_dir>/<split>.txt 的圖片 ID。

    有些清單（例如 Layout）一行有多欄，因此只取第一欄並去除重複、保留原順序。
    """
    path = Path(root) / f"VOC{year}" / "ImageSets" / task_dir / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"找不到切分清單：{path}")
    ids: List[str] = []
    seen = set()
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] not in seen:
            seen.add(parts[0])
            ids.append(parts[0])
    return ids


@dataclass(frozen=True)
class VOCObjects:
    """一張圖的物件標註。"""

    boxes: torch.Tensor  # (N, 4) float32，XYXY，0-based 連續座標
    labels: torch.Tensor  # (N,) int64，1..20
    difficult: torch.Tensor  # (N,) bool


def parse_voc_xml(path: str | Path) -> VOCObjects:
    """解析 VOC XML 標註。

    XML 的座標是從 1 開始的像素編號；轉成連續座標時，左上角減 1、右下角不變，
    這樣 (xmin, ymin, xmax, ymax) 才剛好包住整個像素範圍。
    """
    root = ET.parse(path).getroot()
    boxes, labels, difficult = [], [], []
    for obj in root.iter("object"):
        name = obj.findtext("name", default="").strip()
        if name not in CLASS_TO_INDEX:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1, y1, x2, y2 = (float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([x1 - 1.0, y1 - 1.0, x2, y2])
        labels.append(CLASS_TO_INDEX[name])
        difficult.append(obj.findtext("difficult", default="0").strip() == "1")
    return VOCObjects(
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        labels=torch.tensor(labels, dtype=torch.int64),
        difficult=torch.tensor(difficult, dtype=torch.bool),
    )


def voc_colormap(num_colors: int = 256) -> torch.Tensor:
    """VOC 官方調色盤，回傳 (num_colors, 3) uint8，用於分割結果視覺化。"""
    cmap = torch.zeros(num_colors, 3, dtype=torch.uint8)
    for i in range(num_colors):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = torch.tensor([r, g, b], dtype=torch.uint8)
    return cmap
