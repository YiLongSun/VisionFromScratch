"""語意分割評估：以混淆矩陣計算 mIoU、像素準確率與平均類別準確率。"""

from __future__ import annotations

from typing import Dict, Sequence

import torch


class SegmentationEvaluator:
    def __init__(self, class_names: Sequence[str], ignore_index: int = 255) -> None:
        """class_names 依類別編號 0..N-1 排列（含背景）。"""
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)

    def reset(self) -> None:
        self.confusion.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """preds、targets：(B, H, W) 整數類別。在 GPU 上用 bincount 累計，速度快。"""
        valid = targets != self.ignore_index
        index = targets[valid].long() * self.num_classes + preds[valid].long()
        counts = torch.bincount(index, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self) -> Dict[str, float]:
        conf = self.confusion.double()
        true_positive = conf.diag()
        gt_total = conf.sum(dim=1)  # 每類的真實像素數（列：GT）
        pred_total = conf.sum(dim=0)  # 每類的預測像素數（欄：預測）
        iou = true_positive / (gt_total + pred_total - true_positive)
        class_acc = true_positive / gt_total
        metrics = {
            "mIoU": float(torch.nanmean(iou)),
            "pixel_acc": float(true_positive.sum() / conf.sum().clamp(min=1)),
            "mAcc": float(torch.nanmean(class_acc)),
        }
        for name, value in zip(self.class_names, iou.tolist()):
            metrics[f"IoU/{name}"] = value
        return metrics
