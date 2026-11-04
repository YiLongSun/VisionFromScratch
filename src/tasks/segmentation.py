"""語意分割任務。"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data import (
    CLASS_NAMES_WITH_BG,
    IGNORE_INDEX,
    SegmentationCollate,
    VOCSegmentationDataset,
    build_segmentation_transforms,
)
from ..data.voc import voc_colormap
from ..metrics import SegmentationEvaluator
from ..models import build_model
from .base import TASKS, Task


@TASKS.register("semantic_seg")
class SegmentationTask(Task):
    default_monitor = "val/mIoU"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._colormap = voc_colormap()

    def build_model(self) -> nn.Module:
        return build_model(self.cfg)

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        data = self.cfg.data
        train = VOCSegmentationDataset(
            data.root,
            [(s.year, s.split) for s in data.train_sets],
            transforms=build_segmentation_transforms(data.augment, train=True),
        )
        val = VOCSegmentationDataset(
            data.root,
            [(s.year, s.split) for s in data.val_sets],
            transforms=build_segmentation_transforms(data.augment, train=False),
        )
        return train, val

    @property
    def collate_fn(self) -> Callable:
        return SegmentationCollate(size_divisor=32)

    def training_step(self, model: nn.Module, batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        # loss 一律用 float32 計算，避免 fp16 下 softmax 數值不穩
        loss = F.cross_entropy(logits.float(), masks, ignore_index=IGNORE_INDEX)
        return {"loss": loss}

    @torch.inference_mode()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        autocast: Callable[[], AbstractContextManager],
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        evaluator = SegmentationEvaluator(CLASS_NAMES_WITH_BG, ignore_index=IGNORE_INDEX)
        model.eval()
        for i, (images, masks) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast():
                logits = model(images)
            evaluator.update(logits.argmax(dim=1), masks)
        return evaluator.compute()

    def _colorize(self, mask: torch.Tensor) -> torch.Tensor:
        return self._colormap[mask.long().cpu()].permute(2, 0, 1)

    @torch.inference_mode()
    def visualize(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        autocast: Callable[[], AbstractContextManager],
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """每張輸出「原圖 | 標註 | 預測」並排的圖。"""
        model.eval()
        results: Dict[str, torch.Tensor] = {}
        for i in range(min(num_samples, len(dataset))):
            image, mask = dataset[i]
            image = image.as_subclass(torch.Tensor)
            with autocast():
                logits = model(image.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1)[0]
            canvas = (image * 255).to(torch.uint8)
            results[f"val_vis/{i}"] = torch.cat(
                [canvas, self._colorize(mask.as_subclass(torch.Tensor)), self._colorize(pred)], dim=2
            )
        return results
