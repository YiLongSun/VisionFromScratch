"""Neck：把 backbone 的特徵整理成 head 需要的形式。

- fpn：多尺度 CNN 特徵 → 自頂向下融合，每層通道數相同（Lin et al., 2017）
- simple_pyramid：ViT 的單一尺度特徵 → 用反卷積／池化做出多尺度（ViTDet, Li et al., 2022）
- none：原封不動傳下去（分割 head 自己處理多尺度）

所有 neck 的輸出都是 OrderedDict[str, Tensor]，並提供：
    channels：每層通道數
    strides：每層相對於輸入影像的縮小倍率
    base_names：不含額外層（P6/P7、maxpool）的層名稱，Faster R-CNN 的 RoIAlign 只用這些層
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import Registry

NECKS = Registry("neck")


def _level_name(stride: float) -> str:
    """依 stride 命名，例如 stride 8 → p3。"""
    return f"p{int(round(math.log2(stride)))}"


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(32, channels), channels)


class ExtraLevels(nn.Module):
    """在最後一層之後再加更粗的層。

    maxpool：P6 = max_pool(P5)，torchvision Faster R-CNN 的作法（只給 RPN）
    p6p7：P6 = conv(P5)、P7 = conv(relu(P6))，RetinaNet / FCOS 的作法
    """

    def __init__(self, mode: str, channels: int) -> None:
        super().__init__()
        if mode not in {"none", "maxpool", "p6p7"}:
            raise ValueError(f"未知的 extra_levels：{mode}")
        self.mode = mode
        self.num_levels = {"none": 0, "maxpool": 1, "p6p7": 2}[mode]
        if mode == "p6p7":
            self.p6 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            self.p7 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            for conv in (self.p6, self.p7):
                nn.init.kaiming_uniform_(conv.weight, a=1)
                nn.init.zeros_(conv.bias)

    def forward(self, last: torch.Tensor) -> List[torch.Tensor]:
        if self.mode == "maxpool":
            return [F.max_pool2d(last, kernel_size=1, stride=2)]
        if self.mode == "p6p7":
            p6 = self.p6(last)
            return [p6, self.p7(F.relu(p6))]
        return []


class _PyramidNeck(nn.Module):
    """兩種 FPN 類 neck 的共同部分：命名、額外層與輸出格式。"""

    def __init__(self, base_strides: Sequence[float], out_channels: int, extra_levels: str) -> None:
        super().__init__()
        self.extra = ExtraLevels(extra_levels, out_channels)
        strides = list(base_strides)
        for _ in range(self.extra.num_levels):
            strides.append(strides[-1] * 2)
        self.strides: List[float] = strides
        self.names: List[str] = [_level_name(s) for s in strides]
        self.base_names: List[str] = self.names[: len(base_strides)]
        self.channels: List[int] = [out_channels] * len(strides)
        self.out_channels = out_channels  # torchvision detector 需要這個屬性

    def _pack(self, base: List[torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
        levels = base + self.extra(base[-1])
        return OrderedDict(zip(self.names, levels))


@NECKS.register("fpn")
class FPN(_PyramidNeck):
    """Feature Pyramid Network。"""

    def __init__(
        self,
        in_channels: Sequence[int],
        in_strides: Sequence[float],
        out_channels: int = 256,
        extra_levels: str = "none",
        **_: object,
    ) -> None:
        super().__init__(in_strides, out_channels, extra_levels)
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.output = nn.ModuleList(nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels)
        for conv in [*self.lateral, *self.output]:
            nn.init.kaiming_uniform_(conv.weight, a=1)
            nn.init.zeros_(conv.bias)

    def forward(self, features: List[torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
        # 由最粗的一層往回融合：上採樣後與 lateral 相加
        laterals = [conv(f) for conv, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        outputs = [conv(x) for conv, x in zip(self.output, laterals)]
        return self._pack(outputs)


@NECKS.register("simple_pyramid")
class SimpleFeaturePyramid(_PyramidNeck):
    """ViTDet 的 Simple Feature Pyramid：只用 ViT 最後一層特徵產生多尺度。

    scales 中每個值代表一層：4 → 反卷積上採樣 4 倍、2 → 2 倍、1 → 原尺寸、0.5 → 池化縮小一半。
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        in_strides: Sequence[float],
        out_channels: int = 256,
        extra_levels: str = "none",
        scales: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
        **_: object,
    ) -> None:
        source_stride = in_strides[-1]
        super().__init__([source_stride / s for s in scales], out_channels, extra_levels)
        dim = in_channels[-1]
        self.stages = nn.ModuleList(self._make_stage(dim, s, out_channels) for s in scales)

    @staticmethod
    def _make_stage(dim: int, scale: float, out_channels: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        channels = dim
        if scale == 4.0:
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
                _group_norm(dim // 2),
                nn.GELU(),
                nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2),
            ]
            channels = dim // 4
        elif scale == 2.0:
            layers.append(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2))
            channels = dim // 2
        elif scale == 0.5:
            layers.append(nn.MaxPool2d(2, stride=2))
        elif scale != 1.0:
            raise ValueError(f"simple_pyramid 只支援 scale 4 / 2 / 1 / 0.5，收到 {scale}")
        layers += [
            nn.Conv2d(channels, out_channels, 1, bias=False),
            _group_norm(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
        ]
        return nn.Sequential(*layers)

    def forward(self, features: List[torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
        source = features[-1]
        return self._pack([stage(source) for stage in self.stages])


@NECKS.register("none")
class IdentityNeck(nn.Module):
    """不做任何處理，直接把 backbone 特徵轉成 OrderedDict。"""

    def __init__(self, in_channels: Sequence[int], in_strides: Sequence[float], **_: object) -> None:
        super().__init__()
        self.channels = list(in_channels)
        self.strides = list(in_strides)
        self.names = [f"c{i}" for i in range(len(in_channels))]
        self.base_names = list(self.names)

    def forward(self, features: List[torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict(zip(self.names, features))


class FeatureExtractor(nn.Module):
    """backbone + neck；torchvision detector 會把它當成 backbone 使用。"""

    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.channels: List[int] = neck.channels
        self.strides: List[float] = neck.strides
        self.names: List[str] = neck.names
        self.base_names: List[str] = neck.base_names
        if len(set(self.channels)) == 1:
            self.out_channels = self.channels[0]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.neck(self.backbone(x))
