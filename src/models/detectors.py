"""物件偵測模型：以 torchvision 的偵測架構為 head，接上任意 backbone + neck。

torchvision detector 只要求 backbone 回傳 OrderedDict 特徵並有 out_channels 屬性，
因此 FeatureExtractor 可以直接放進去。anchor 尺寸依每層 stride 自動計算，換 backbone 不必手動調整。

所有 detector 都會在內部做：
    訓練：隨機選一個 min_size 縮放 → 正規化 → 補齊成 batch
    推論：縮放 → 正規化 → 預測 → 框座標還原回原圖尺寸
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from torch import nn
from torchvision.models.detection import FCOS, FasterRCNN, RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from ..registry import Registry
from .necks import FeatureExtractor

DETECTORS = Registry("detector")


def _check_levels(features: FeatureExtractor, name: str) -> None:
    if not hasattr(features, "out_channels"):
        raise ValueError(f"{name} 需要每層通道數相同的特徵，請把 neck 設為 fpn 或 simple_pyramid")


@DETECTORS.register("faster_rcnn")
def build_faster_rcnn(
    features: FeatureExtractor, num_classes: int, transform_kwargs: Dict[str, Any], **kwargs: Any
) -> nn.Module:
    """兩階段：RPN 提出候選框 → RoIAlign → 分類與框回歸。anchor 邊長為 stride × 8。"""
    _check_levels(features, "faster_rcnn")
    sizes = tuple((int(8 * s),) for s in features.strides)
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    roi_pool = MultiScaleRoIAlign(featmap_names=features.base_names, output_size=7, sampling_ratio=2)
    return FasterRCNN(
        features,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool,
        **transform_kwargs,
        **kwargs,
    )


@DETECTORS.register("retinanet")
def build_retinanet(
    features: FeatureExtractor, num_classes: int, transform_kwargs: Dict[str, Any], **kwargs: Any
) -> nn.Module:
    """單階段 anchor-based，Focal Loss。每層 3 種尺寸 × 3 種長寬比，基準邊長為 stride × 4。"""
    _check_levels(features, "retinanet")
    sizes = tuple(tuple(int(4 * s * 2 ** (i / 3)) for i in range(3)) for s in features.strides)
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    return RetinaNet(features, num_classes=num_classes, anchor_generator=anchor_generator, **transform_kwargs, **kwargs)


@DETECTORS.register("fcos")
def build_fcos(
    features: FeatureExtractor, num_classes: int, transform_kwargs: Dict[str, Any], **kwargs: Any
) -> nn.Module:
    """單階段 anchor-free：每個位置直接預測到四邊的距離。anchor 只用來標示位置，邊長等於 stride。"""
    _check_levels(features, "fcos")
    sizes = tuple((int(s),) for s in features.strides)
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=((1.0,),) * len(sizes))
    return FCOS(features, num_classes=num_classes, anchor_generator=anchor_generator, **transform_kwargs, **kwargs)


def detector_transform_kwargs(
    min_size: Sequence[int], max_size: int, mean: Tuple[float, ...], std: Tuple[float, ...]
) -> Dict[str, Any]:
    """GeneralizedRCNNTransform 的參數：縮放尺寸與 backbone 對應的正規化。"""
    return {
        "min_size": tuple(min_size),
        "max_size": max_size,
        "image_mean": list(mean),
        "image_std": list(std),
    }
