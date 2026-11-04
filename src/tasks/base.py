"""Task 介面：把「任務相關」的部分從 Trainer 抽離。

Trainer 只負責通用的訓練流程（AMP、梯度累積、存檔、log），
資料集、模型、loss、評估與視覺化則由各 Task 實作。新增任務（例如實例分割）時只要新增一個 Task。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from ..registry import Registry

TASKS = Registry("task")


class Task(ABC):
    #: 訓練中預設監控的指標（可被 checkpoint.monitor 覆寫）
    default_monitor: str = ""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    # ---- 建構 ----
    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def build_datasets(self) -> Tuple[Dataset, Dataset]:
        """回傳 (訓練集, 驗證集)。"""

    @property
    @abstractmethod
    def collate_fn(self) -> Callable: ...

    # ---- 訓練與評估 ----
    @abstractmethod
    def training_step(self, model: nn.Module, batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
        """回傳 loss dict，其中 "loss" 為要 backward 的總 loss。"""

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        autocast: Callable[[], AbstractContextManager],
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]: ...

    @abstractmethod
    def visualize(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        autocast: Callable[[], AbstractContextManager],
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """回傳 {tag: uint8 影像 (3, H, W)}，寫入 TensorBoard。"""


def build_task(cfg: ExperimentConfig) -> Task:
    return TASKS.build(cfg.experiment.task, cfg)
