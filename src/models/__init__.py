"""模型組裝：依設定把 backbone、neck、head 串起來。"""

from __future__ import annotations

from torch import nn

from ..config import ExperimentConfig
from .backbone import TimmBackbone, is_vit
from .detectors import DETECTORS, detector_transform_kwargs
from .necks import NECKS, FeatureExtractor
from .segmentors import SEG_HEADS, SegmentationModel

__all__ = ["DETECTORS", "NECKS", "SEG_HEADS", "TimmBackbone", "build_model", "build_feature_extractor"]


def build_feature_extractor(cfg: ExperimentConfig) -> FeatureExtractor:
    backbone = TimmBackbone(cfg.model.backbone)
    neck_cfg = cfg.model.neck
    neck = NECKS.build(
        neck_cfg.name,
        in_channels=backbone.channels,
        in_strides=backbone.strides,
        out_channels=neck_cfg.out_channels,
        extra_levels=neck_cfg.extra_levels,
        scales=tuple(neck_cfg.scales),
    )
    return FeatureExtractor(backbone, neck)


def build_model(cfg: ExperimentConfig) -> nn.Module:
    """依 experiment.task 建立偵測或分割模型。"""
    features = build_feature_extractor(cfg)
    backbone: TimmBackbone = features.backbone  # type: ignore[assignment]
    head_cfg = cfg.model.head

    if cfg.experiment.task == "detection":
        aug = cfg.data.augment
        transform_kwargs = detector_transform_kwargs(aug.min_size, aug.max_size, backbone.mean, backbone.std)
        return DETECTORS.build(
            head_cfg.name,
            features,
            num_classes=head_cfg.num_classes,
            transform_kwargs=transform_kwargs,
            **head_cfg.kwargs,
        )

    head = SEG_HEADS.build(
        head_cfg.name, in_channels=features.channels, num_classes=head_cfg.num_classes, **head_cfg.kwargs
    )
    # ViT 的 dynamic_img_pad 會自己補邊，CNN 則補到最大 stride 的倍數
    size_divisor = 1 if is_vit(cfg.model.backbone.name) else int(max(features.strides))
    return SegmentationModel(features, head, mean=backbone.mean, std=backbone.std, size_divisor=size_divisor)
