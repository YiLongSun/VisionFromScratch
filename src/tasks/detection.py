"""物件偵測任務。"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import draw_bounding_boxes

from ..data import VOC_CLASSES, VOCDetectionDataset, build_detection_transforms, detection_collate
from ..metrics import VOCDetectionEvaluator
from ..models import build_model
from .base import TASKS, Task

_VIS_SCORE_THRESHOLD = 0.5


@contextmanager
def _test_resolution(model: nn.Module, min_size: int) -> Iterator[None]:
    """推論時改用固定的 test_min_size；torchvision 推論預設取 min_size 的最後一個值。"""
    transform = model.transform
    original = transform.min_size
    transform.min_size = (min_size,)
    try:
        yield
    finally:
        transform.min_size = original


def _to_device_targets(targets: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """只留 detector 需要的欄位，並把 tv_tensors 轉回一般 Tensor。"""
    return [
        {
            "boxes": t["boxes"].as_subclass(torch.Tensor).to(device, non_blocking=True),
            "labels": t["labels"].to(device, non_blocking=True),
        }
        for t in targets
    ]


@TASKS.register("detection")
class DetectionTask(Task):
    default_monitor = "val/mAP50"

    def build_model(self) -> nn.Module:
        return build_model(self.cfg)

    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        data = self.cfg.data
        train = VOCDetectionDataset(
            data.root,
            [(s.year, s.split) for s in data.train_sets],
            transforms=build_detection_transforms(data.augment, train=True),
            use_difficult=data.use_difficult,
        )
        # 評估集保留 difficult 物件，由 evaluator 依 VOC 規則忽略
        val = VOCDetectionDataset(
            data.root,
            [(s.year, s.split) for s in data.val_sets],
            transforms=build_detection_transforms(data.augment, train=False),
            use_difficult=True,
        )
        return train, val

    @property
    def collate_fn(self) -> Callable:
        return detection_collate

    def training_step(self, model: nn.Module, batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
        images, targets = batch
        images = [img.as_subclass(torch.Tensor).to(device, non_blocking=True) for img in images]
        loss_dict = model(images, _to_device_targets(targets, device))
        return {"loss": sum(loss_dict.values()), **loss_dict}

    @torch.inference_mode()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        autocast: Callable[[], AbstractContextManager],
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        evaluator = VOCDetectionEvaluator(VOC_CLASSES)
        model.eval()
        with _test_resolution(model, self.cfg.data.augment.test_min_size):
            for i, (images, targets) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                images = [img.as_subclass(torch.Tensor).to(device, non_blocking=True) for img in images]
                with autocast():
                    outputs = model(images)
                evaluator.update(outputs, targets)
        return evaluator.compute()

    @torch.inference_mode()
    def visualize(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        autocast: Callable[[], AbstractContextManager],
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        model.eval()
        results: Dict[str, torch.Tensor] = {}
        with _test_resolution(model, self.cfg.data.augment.test_min_size):
            for i in range(min(num_samples, len(dataset))):
                image, _ = dataset[i]
                image = image.as_subclass(torch.Tensor)
                with autocast():
                    output = model([image.to(device)])[0]
                keep = output["scores"] >= _VIS_SCORE_THRESHOLD
                labels = [
                    f"{VOC_CLASSES[int(c) - 1]} {s:.2f}"
                    for c, s in zip(output["labels"][keep].tolist(), output["scores"][keep].tolist())
                ]
                canvas = (image * 255).to(torch.uint8)
                results[f"val_vis/{i}"] = draw_bounding_boxes(
                    canvas, output["boxes"][keep].float().cpu(), labels=labels, colors="lime", width=2
                )
        return results
