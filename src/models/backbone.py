"""timm backbone 包裝。

任何 timm 支援 features_only 的模型都能使用，包括：
- CNN：resnet50、convnext_tiny、efficientnet_b3、mobilenetv3_large_100 …
- ViT：vit_small_patch16_224.dino、vit_small_patch14_dinov2.lvd142m、vit_small_patch16_dinov3.lvd1689m …
也能透過 checkpoint 載入 timm 沒有的外部權重（例如 DINO ResNet-50）。

輸出為多尺度特徵 list；channels 與 strides 屬性讓後面的 neck / head 自動對齊。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import timm
import torch
from timm.layers import freeze_batch_norm_2d
from timm.models import get_arch_name, is_model_in_modules
from torch import nn

from ..config import BackboneSection
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 這些 timm 模組裡的模型是 ViT 類（單一尺度、patch 輸入）
_VIT_MODULES = ["vision_transformer", "vision_transformer_hybrid", "eva", "deit", "beit"]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_STRIP_PREFIXES = ("module.", "backbone.", "encoder.")


def is_vit(model_name: str) -> bool:
    return is_model_in_modules(get_arch_name(model_name), _VIT_MODULES)


class TimmBackbone(nn.Module):
    """以 timm features_only 模式建立的 backbone。"""

    def __init__(self, cfg: BackboneSection) -> None:
        super().__init__()
        kwargs = dict(cfg.kwargs)
        if is_vit(cfg.name):
            # 讓 ViT 接受任意尺寸輸入：位置編碼自動內插，尺寸不是 patch 倍數時自動補邊
            kwargs.setdefault("dynamic_img_size", True)
            kwargs.setdefault("dynamic_img_pad", True)

        use_timm_weights = cfg.pretrained and cfg.checkpoint is None
        self.body = timm.create_model(
            cfg.name,
            pretrained=use_timm_weights,
            features_only=True,
            out_indices=tuple(cfg.out_indices),
            **kwargs,
        )
        if cfg.checkpoint is not None:
            load_external_weights(self.body, cfg.checkpoint, cfg.checkpoint_key)

        self.channels: List[int] = list(self.body.feature_info.channels())
        self.strides: List[int] = list(self.body.feature_info.reduction())

        pretrained_cfg: Dict = getattr(self.body, "pretrained_cfg", None) or {}
        self.mean = tuple(cfg.mean or pretrained_cfg.get("mean", _IMAGENET_MEAN))
        self.std = tuple(cfg.std or pretrained_cfg.get("std", _IMAGENET_STD))

        if cfg.freeze_bn:
            self.body = freeze_batch_norm_2d(self.body)
        if cfg.freeze:
            self.body.requires_grad_(False)
        self.frozen = cfg.freeze

        logger.info(
            "Backbone %s：channels=%s strides=%s mean=%s",
            cfg.name,
            self.channels,
            self.strides,
            tuple(round(m, 3) for m in self.mean),
        )

    def train(self, mode: bool = True) -> "TimmBackbone":
        # 凍結的 backbone 永遠維持 eval 模式（BatchNorm 統計量、dropout 都不變）
        super().train(mode and not self.frozen)
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.body(x))


def load_external_weights(module: nn.Module, source: str, key: Optional[str] = None) -> None:
    """載入外部權重到 timm features_only 模型。

    會自動處理常見的 key 前綴差異（module.、backbone.），以及 timm 包裝層多出的 "model." 前綴；
    形狀不符或用不到的 key（例如分類頭 fc）會略過並記錄數量。
    """
    if source.startswith(("http://", "https://")):
        state = torch.hub.load_state_dict_from_url(source, map_location="cpu", weights_only=False)
    else:
        # 外部 checkpoint 可能含有非 tensor 物件（如 argparse.Namespace），只載入你信任的檔案
        state = torch.load(source, map_location="cpu", weights_only=False)
    if key is not None:
        state = state[key]

    target = module.state_dict()
    matched: Dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        clean = raw_key
        for prefix in _STRIP_PREFIXES:
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        for candidate in (clean, f"model.{clean}"):
            if candidate in target and target[candidate].shape == value.shape:
                matched[candidate] = value
                break

    if not matched:
        raise RuntimeError(f"外部權重 {source} 沒有任何參數能對應到模型，請檢查 checkpoint_key 或模型名稱")
    missing = [k for k in target if k not in matched and not k.endswith("num_batches_tracked")]
    module.load_state_dict(matched, strict=False)
    logger.info("外部權重載入：%d 個參數成功，%d 個未載入（模型端）", len(matched), len(missing))
    if missing:
        logger.warning("未載入的參數（前 10 個）：%s", missing[:10])
