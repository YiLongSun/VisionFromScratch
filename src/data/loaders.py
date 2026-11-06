"""Collate 函式與 DataLoader 建構。"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .sampling import ResumableRandomSampler, SeededDataset
from .voc import IGNORE_INDEX


def detection_collate(batch):
    """偵測：影像尺寸不一，保持 list，交給 detector 自己 resize / pad。"""
    images, targets = zip(*batch)
    return list(images), list(targets)


class SegmentationCollate:
    """分割：把一個 batch 補齊到相同尺寸（且為 size_divisor 的倍數），mask 補 IGNORE_INDEX。"""

    def __init__(self, size_divisor: int = 32) -> None:
        self.size_divisor = size_divisor

    def _round_up(self, value: int) -> int:
        return int(math.ceil(value / self.size_divisor) * self.size_divisor)

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images, masks = zip(*batch)
        height = self._round_up(max(img.shape[-2] for img in images))
        width = self._round_up(max(img.shape[-1] for img in images))
        padded_images: List[torch.Tensor] = []
        padded_masks: List[torch.Tensor] = []
        for img, mask in zip(images, masks):
            pad = (0, width - img.shape[-1], 0, height - img.shape[-2])  # 只補右邊與下邊
            padded_images.append(F.pad(img.as_subclass(torch.Tensor), pad, value=0.0))
            padded_masks.append(F.pad(mask.as_subclass(torch.Tensor).long(), pad, value=IGNORE_INDEX))
        return torch.stack(padded_images), torch.stack(padded_masks)


def build_train_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    seed: int,
    pin_memory: bool = True,
) -> Tuple[DataLoader, ResumableRandomSampler]:
    """訓練用 DataLoader；回傳 sampler 讓 Trainer 控制 epoch 與續訓位置。"""
    sampler = ResumableRandomSampler(len(dataset), seed=seed, shuffle=True)  # type: ignore[arg-type]
    loader = DataLoader(
        SeededDataset(dataset),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # 避免最後一個 batch 只有 1 張，BatchNorm 會出錯
        persistent_workers=num_workers > 0,
        # 獨立的 generator：建立 iterator 時 DataLoader 會抽一個亂數當 worker 種子，
        # 若用全域 RNG，續訓時抽取時機不同，會讓之後 dropout 等模型隨機性與不中斷時不一致
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, sampler


def build_val_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    pin_memory: bool = True,
) -> DataLoader:
    """評估用 DataLoader：固定順序、不丟棄最後一個 batch。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        generator=torch.Generator().manual_seed(0),  # 同上，不動到全域 RNG
    )


def num_batches_for(max_samples: Optional[int], batch_size: int) -> Optional[int]:
    """把「最多評估幾張圖」換算成 batch 數。"""
    return None if max_samples is None else max(1, math.ceil(max_samples / batch_size))
