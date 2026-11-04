"""零散的小工具。"""

from __future__ import annotations

import torch
from torch import nn


def resolve_device(name: str) -> torch.device:
    """auto 時有 GPU 就用 GPU。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """回傳 (全部參數量, 可訓練參數量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
