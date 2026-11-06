"""資料增強（torchvision.transforms.v2）。

v2 可以同時轉換影像、bounding box 與 mask，偵測與分割共用同一套寫法。
輸出影像為 float32、範圍 [0, 1]；正規化（減 mean 除 std）在模型內部做，
這樣換 backbone（例如 DINO 與 ImageNet 權重的 mean/std 不同）時不必改資料管線。
"""

from __future__ import annotations

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..config import AugmentSection
from .voc import IGNORE_INDEX


def _to_float_image() -> v2.Transform:
    # 只轉換影像；mask 保持整數
    return v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True)


def _detection_labels_getter(inputs):
    """SanitizeBoundingBoxes 移除框時，要一併移除的欄位（寫成函式才能被 pickle 給 worker）。"""
    target = inputs[1]
    return target["labels"], target["difficult"]


def build_detection_transforms(aug: AugmentSection, train: bool) -> v2.Compose:
    """偵測用 transforms。縮放交由 torchvision detector 內部的 GeneralizedRCNNTransform 處理。"""
    ops: list[v2.Transform] = []
    if train:
        if aug.photometric:
            ops.append(v2.RandomPhotometricDistort(p=0.5))
        if aug.hflip_prob > 0:
            ops.append(v2.RandomHorizontalFlip(p=aug.hflip_prob))
        # 移除增強後退化（寬或高為 0）的框，並同步移除對應的 labels 與 difficult
        ops.append(v2.SanitizeBoundingBoxes(labels_getter=_detection_labels_getter))
    ops.append(_to_float_image())
    return v2.Compose(ops)


def build_segmentation_transforms(aug: AugmentSection, train: bool) -> v2.Compose:
    """分割用 transforms：隨機縮放 → 隨機裁切（不足處補 0 / 255）→ 翻轉 → 色彩擾動。"""
    ops: list[v2.Transform] = []
    if train:
        low, high = aug.scale_range
        ops += [
            v2.RandomResize(min_size=int(aug.base_size * low), max_size=int(aug.base_size * high)),
            v2.RandomCrop(
                aug.crop_size,
                pad_if_needed=True,
                fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE_INDEX},
            ),
        ]
        if aug.hflip_prob > 0:
            ops.append(v2.RandomHorizontalFlip(p=aug.hflip_prob))
        if aug.photometric:
            ops.append(v2.RandomPhotometricDistort(p=0.5))
    ops.append(_to_float_image())
    return v2.Compose(ops)
