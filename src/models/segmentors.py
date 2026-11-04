"""語意分割模型：backbone + neck + 分割 head。

- deeplabv3plus：ASPP 擷取多尺度語境 + 低階特徵還原邊緣（Chen et al., 2018）
- fpn_seg：Semantic FPN，各層上採樣到同一尺寸後相加（Kirillov et al., 2019）
- linear：BatchNorm + 1×1 卷積，常用於評估自監督特徵（例如 DINOv2 linear probe）

head 的輸出解析度低於輸入，SegmentationModel 會把 logits 上採樣回輸入尺寸。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import Registry
from .necks import FeatureExtractor

SEG_HEADS = Registry("segmentation head")


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1) -> nn.Sequential:
    padding = 0 if kernel == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling：1×1、三種 dilation 的 3×3、全域平均池化，共 5 個分支。"""

    def __init__(self, in_ch: int, out_ch: int, rates: Sequence[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [_conv_bn_relu(in_ch, out_ch, kernel=1)] + [_conv_bn_relu(in_ch, out_ch, dilation=r) for r in rates]
        )
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), _conv_bn_relu(in_ch, out_ch, kernel=1))
        self.project = nn.Sequential(
            _conv_bn_relu(out_ch * (len(rates) + 2), out_ch, kernel=1),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        pooled = F.interpolate(self.image_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([*outs, pooled], dim=1))


@SEG_HEADS.register("deeplabv3plus")
class DeepLabV3PlusHead(nn.Module):
    """使用第一層特徵當低階特徵、最後一層特徵給 ASPP。"""

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        aspp_channels: int = 256,
        aspp_rates: Sequence[int] = (6, 12, 18),
        low_level_channels: int = 48,
        **_: object,
    ) -> None:
        super().__init__()
        self.aspp = ASPP(in_channels[-1], aspp_channels, aspp_rates)
        self.low_level = _conv_bn_relu(in_channels[0], low_level_channels, kernel=1)
        self.decoder = nn.Sequential(
            _conv_bn_relu(aspp_channels + low_level_channels, 256),
            _conv_bn_relu(256, 256),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        low = self.low_level(features[0])
        high = F.interpolate(self.aspp(features[-1]), size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.decoder(torch.cat([high, low], dim=1)))


@SEG_HEADS.register("fpn_seg")
class SemanticFPNHead(nn.Module):
    """每層先 3×3 卷積，再上採樣到第一層尺寸相加，最後分類。適合接 fpn 或 simple_pyramid neck。"""

    def __init__(self, in_channels: Sequence[int], num_classes: int, channels: int = 128, **_: object) -> None:
        super().__init__()
        self.scale_heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, channels, 3, padding=1, bias=False),
                nn.GroupNorm(math.gcd(32, channels), channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        )
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(channels, num_classes, 1))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        size = features[0].shape[-2:]
        fused = sum(
            F.interpolate(head(f), size=size, mode="bilinear", align_corners=False)
            for head, f in zip(self.scale_heads, features)
        )
        return self.classifier(fused)


@SEG_HEADS.register("linear")
class LinearHead(nn.Module):
    """把所有特徵上採樣到第一層尺寸後串接，經 BatchNorm 與 1×1 卷積分類。"""

    def __init__(self, in_channels: Sequence[int], num_classes: int, **_: object) -> None:
        super().__init__()
        total = sum(in_channels)
        self.norm = nn.BatchNorm2d(total)
        self.classifier = nn.Conv2d(total, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        size = features[0].shape[-2:]
        x = torch.cat(
            [
                f if f.shape[-2:] == size else F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                for f in features
            ],
            dim=1,
        )
        return self.classifier(self.norm(x))


class SegmentationModel(nn.Module):
    """完整的分割模型：輸入 [0, 1] 影像，輸出與輸入同尺寸的 logits。"""

    def __init__(
        self,
        features: FeatureExtractor,
        head: nn.Module,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
        size_divisor: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = features  # 命名為 backbone 與 torchvision detector 一致
        self.head = head
        self.size_divisor = size_divisor
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        x = (images - self.mean) / self.std
        # 補到 size_divisor 的倍數，避免各層特徵尺寸對不齊
        pad_h = (-height) % self.size_divisor
        pad_w = (-width) % self.size_divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        features = list(self.backbone(x).values())
        logits = self.head(features)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits[..., :height, :width]
