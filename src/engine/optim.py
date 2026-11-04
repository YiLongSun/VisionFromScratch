"""Optimizer 與學習率排程的建構。"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..config import OptimSection, SchedulerSection
from ..models.backbone import TimmBackbone

# 名稱含這些字串的參數不做 weight decay（ViT 的位置編碼與特殊 token）
_NO_DECAY_KEYWORDS = ("pos_embed", "cls_token", "reg_token", "mask_token", "rope")


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def build_param_groups(model: nn.Module, cfg: OptimSection) -> List[Dict]:
    """把參數分成最多 4 組：{backbone, 其他} × {要 decay, 不 decay}。

    每組都帶有 name 欄位，TensorBoard 會依此分別記錄學習率。
    """
    backbone_ids = {id(p) for m in model.modules() if isinstance(m, TimmBackbone) for p in m.parameters()}
    groups: Dict[str, Dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        part = "backbone" if id(param) in backbone_ids else "head"
        no_decay = cfg.no_weight_decay_norm_bias and (
            param.ndim <= 1 or name.endswith(".bias") or any(k in name for k in _NO_DECAY_KEYWORDS)
        )
        key = f"{part}{'_no_decay' if no_decay else ''}"
        if key not in groups:
            lr_mult = cfg.backbone_lr_mult if part == "backbone" else 1.0
            groups[key] = {
                "name": key,
                "params": [],
                "lr": cfg.lr * lr_mult,
                "weight_decay": 0.0 if no_decay else cfg.weight_decay,
            }
        groups[key]["params"].append(param)
    if not groups:
        raise ValueError("模型沒有任何可訓練的參數")
    return list(groups.values())


def build_optimizer(model: nn.Module, cfg: OptimSection) -> Optimizer:
    groups = build_param_groups(model, cfg)
    name = cfg.name.lower()
    if name == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    if name == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)
    raise ValueError(f"未知的 optimizer：{cfg.name}（可用 sgd / adam / adamw）")


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def _main_schedule(cfg: SchedulerSection, total_steps: int) -> Callable[[int], float]:
    """warmup 之後的主排程，回傳「step → 學習率倍率」。"""
    warmup = cfg.warmup_steps
    decay_steps = max(1, total_steps - warmup)
    name = cfg.name.lower()

    if name == "constant":
        return lambda step: 1.0
    if name == "cosine":

        def cosine(step: int) -> float:
            progress = min(max(step - warmup, 0) / decay_steps, 1.0)
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return cosine
    if name == "poly":

        def poly(step: int) -> float:
            progress = min(max(step - warmup, 0) / decay_steps, 1.0)
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 - progress) ** cfg.power

        return poly
    if name == "multistep":
        milestones = sorted(int(m * total_steps) for m in cfg.milestones)
        return lambda step: cfg.gamma ** sum(step >= m for m in milestones)
    if name == "step":
        interval = max(1, int(cfg.step_size * total_steps))
        return lambda step: cfg.gamma ** (step // interval)
    raise ValueError(f"未知的 scheduler：{cfg.name}（可用 constant / cosine / poly / multistep / step）")


def build_scheduler(optimizer: Optimizer, cfg: SchedulerSection, total_steps: int) -> LambdaLR:
    """線性 warmup + 主排程。每個 optimizer step 呼叫一次 scheduler.step()。

    使用 LambdaLR 統一實作：倍率套用在每組參數各自的初始學習率上，所以 backbone_lr_mult 會被保留。
    續訓時只要用相同設定重建，再 load_state_dict 還原目前步數即可。
    """
    main = _main_schedule(cfg, total_steps)
    warmup = cfg.warmup_steps
    start = cfg.warmup_start_factor

    def lr_lambda(step: int) -> float:
        factor = main(step)
        if step < warmup:
            factor *= start + (1 - start) * step / warmup
        return factor

    return LambdaLR(optimizer, lr_lambda)
