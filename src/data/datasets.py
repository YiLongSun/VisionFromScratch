"""VOC 偵測與語意分割 Dataset。

兩者都只依照 ImageSets 清單取圖，不會掃描整個 JPEGImages。
回傳的影像為 uint8 的 tv_tensors.Image，交給 transforms v2 處理後再轉成 float。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import pil_to_tensor

from .voc import parse_voc_xml, read_split_ids

Transform = Callable[..., Tuple]


def _load_image(path: Path) -> tv_tensors.Image:
    """以 PIL 讀圖並統一轉成 RGB（torchvision.io 的影像解碼已被標為棄用）。"""
    with Image.open(path) as img:
        return tv_tensors.Image(pil_to_tensor(img.convert("RGB")))


class VOCDetectionDataset(Dataset):
    """VOC 物件偵測資料集，可合併多個年份與切分（例如 07+12 trainval）。

    回傳 (image, target)，target 內容：
        boxes:     tv_tensors.BoundingBoxes (N, 4) XYXY
        labels:    int64 (N,)，1..20
        difficult: bool (N,)
        image_id:  int，資料集內的索引
    """

    def __init__(
        self,
        root: str | Path,
        sets: Sequence[Tuple[str, str]],
        transforms: Optional[Transform] = None,
        use_difficult: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transforms = transforms
        self.use_difficult = use_difficult
        self.samples: List[Tuple[Path, Path]] = []
        for year, split in sets:
            year_dir = self.root / f"VOC{year}"
            for image_id in read_split_ids(self.root, year, "Main", split):
                self.samples.append(
                    (year_dir / "JPEGImages" / f"{image_id}.jpg", year_dir / "Annotations" / f"{image_id}.xml")
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, xml_path = self.samples[index]
        image = _load_image(image_path)
        objects = parse_voc_xml(xml_path)

        keep = torch.ones_like(objects.difficult) if self.use_difficult else ~objects.difficult
        target: Dict[str, object] = {
            "boxes": tv_tensors.BoundingBoxes(objects.boxes[keep], format="XYXY", canvas_size=tuple(image.shape[-2:])),
            "labels": objects.labels[keep],
            "difficult": objects.difficult[keep],
            "image_id": index,
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class VOCSegmentationDataset(Dataset):
    """VOC 語意分割資料集。

    標註 PNG 是調色盤模式，像素值本身就是類別編號（0 背景、1..20 物件、255 邊界忽略）。
    回傳 (image, mask)，mask 為 tv_tensors.Mask (H, W) uint8（縮放時 uint8 才能用最近鄰插值，collate 時再轉 int64）。
    """

    def __init__(
        self,
        root: str | Path,
        sets: Sequence[Tuple[str, str]],
        transforms: Optional[Transform] = None,
    ) -> None:
        self.root = Path(root)
        self.transforms = transforms
        self.samples: List[Tuple[Path, Path]] = []
        for year, split in sets:
            year_dir = self.root / f"VOC{year}"
            for image_id in read_split_ids(self.root, year, "Segmentation", split):
                self.samples.append(
                    (year_dir / "JPEGImages" / f"{image_id}.jpg", year_dir / "SegmentationClass" / f"{image_id}.png")
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = _load_image(image_path)
        with Image.open(mask_path) as png:
            # 不能 convert("RGB")，要直接讀調色盤索引
            mask = tv_tensors.Mask(torch.from_numpy(np.array(png, dtype=np.uint8)))
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask
