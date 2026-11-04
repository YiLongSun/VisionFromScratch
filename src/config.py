"""實驗設定。

以 dataclass 定義完整 schema，透過 OmegaConf 載入「單一 YAML 檔」並支援命令列覆寫：

    python -m src.train --config configs/detection.yaml optim.lr=0.005 data.batch_size=4

YAML 只需寫出想改的欄位，其餘沿用這裡的預設值；打錯欄位名稱或型別不符會直接報錯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

_MISSING = object()


# --------------------------------------------------------------------------- #
# 各區塊設定
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentSection:
    """實驗層級的設定：名稱、任務類型、輸出位置。"""

    name: str = "experiment"
    task: str = "detection"  # detection | semantic_seg
    seed: int = 42
    # 所有輸出（checkpoint、TensorBoard、設定快照）都放在 output_root/name 底下。
    # 在 Colab 上請指向 Google Drive，斷線後才不會遺失。
    output_root: str = "/content/drive/MyDrive/VisionFromScratch/outputs"


@dataclass
class DatasetSplit:
    """一組「年份 + 切分」，例如 VOC2007 trainval。"""

    year: str = "2007"
    split: str = "trainval"


@dataclass
class AugmentSection:
    """資料增強設定；偵測與分割各自使用其中相關的欄位。"""

    hflip_prob: float = 0.5
    photometric: bool = True  # 隨機亮度、對比、飽和度、色相
    # ---- 偵測：縮放交由 torchvision detector 內部的 transform 處理 ----
    min_size: List[int] = field(default_factory=lambda: [480, 512, 544, 576, 608])
    max_size: int = 1000
    test_min_size: int = 544
    # ---- 語意分割 ----
    base_size: int = 512  # 隨機縮放的基準短邊長度
    scale_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    crop_size: int = 480


@dataclass
class DataSection:
    """資料集與 DataLoader 設定。"""

    root: str = "/content/data/VOC0712/VOC_dataset/VOCdevkit"
    train_sets: List[DatasetSplit] = field(default_factory=list)
    val_sets: List[DatasetSplit] = field(default_factory=list)
    use_difficult: bool = False  # 訓練時是否保留 difficult 物件（評估時一律保留並忽略）
    batch_size: int = 8
    val_batch_size: int = 8
    num_workers: int = 2  # Colab 只有 2 顆 CPU
    pin_memory: bool = True
    val_max_samples: Optional[int] = None  # 訓練中的定期評估只取前 N 張以節省時間
    augment: AugmentSection = field(default_factory=AugmentSection)


@dataclass
class BackboneSection:
    """timm backbone 設定。name 可為任何 timm 模型名稱（含權重 tag）。"""

    name: str = "resnet50.a1_in1k"
    pretrained: bool = True
    # 外部權重（例如 DINO ResNet-50），可為本機路徑或 URL；設定後會忽略 pretrained。
    checkpoint: Optional[str] = None
    checkpoint_key: Optional[str] = None  # 權重存在 checkpoint 的哪個 key 底下，例如 "teacher"
    out_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    freeze: bool = False  # 凍結整個 backbone（linear probe 用）
    freeze_bn: bool = True  # 將 BatchNorm 轉為 FrozenBatchNorm（小 batch 時建議開啟）
    mean: Optional[List[float]] = None  # 預設採用權重本身的正規化參數
    std: Optional[List[float]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 直接傳給 timm.create_model


@dataclass
class NeckSection:
    """Neck 設定。"""

    name: str = "fpn"  # fpn | simple_pyramid | none
    out_channels: int = 256
    extra_levels: str = "none"  # none | maxpool | p6p7
    # simple_pyramid 專用：對 ViT 最後一層特徵做的縮放倍率（4 代表上採樣 4 倍）
    scales: List[float] = field(default_factory=lambda: [4.0, 2.0, 1.0, 0.5])


@dataclass
class HeadSection:
    """Head 設定。"""

    name: str = "fcos"  # 偵測：faster_rcnn | retinanet | fcos；分割：deeplabv3plus | fpn_seg | linear
    num_classes: int = 21  # 20 類 + 背景
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSection:
    backbone: BackboneSection = field(default_factory=BackboneSection)
    neck: NeckSection = field(default_factory=NeckSection)
    head: HeadSection = field(default_factory=HeadSection)


@dataclass
class OptimSection:
    """Optimizer 設定。"""

    name: str = "sgd"  # sgd | adam | adamw
    lr: float = 0.01
    weight_decay: float = 1e-4
    momentum: float = 0.9  # sgd
    nesterov: bool = False  # sgd
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])  # adam / adamw
    eps: float = 1e-8
    backbone_lr_mult: float = 1.0  # backbone 學習率 = lr × 此倍率
    no_weight_decay_norm_bias: bool = True  # norm、bias、位置編碼不做 weight decay


@dataclass
class SchedulerSection:
    """學習率排程；所有時間點都以 optimizer step 計算。"""

    name: str = "cosine"  # constant | cosine | multistep | step | poly
    warmup_steps: int = 500
    warmup_start_factor: float = 0.001
    min_lr_ratio: float = 0.0  # cosine / poly 的最低學習率（相對於初始值）
    milestones: List[float] = field(default_factory=lambda: [0.67, 0.89])  # multistep，總 step 的比例
    step_size: float = 0.33  # step，總 step 的比例
    gamma: float = 0.1  # multistep / step 的衰減倍率
    power: float = 0.9  # poly


@dataclass
class TrainSection:
    """訓練迴圈設定。max_steps 與 max_epochs 擇一，max_steps 優先。"""

    max_steps: Optional[int] = None
    max_epochs: Optional[int] = 12
    grad_accum_steps: int = 1
    grad_clip_norm: Optional[float] = 10.0
    precision: str = "fp16"  # fp16 | bf16 | fp32（T4 請用 fp16）
    log_every_steps: int = 20
    eval_every_steps: int = 2000
    cudnn_benchmark: bool = False  # 輸入尺寸固定（分割）時開啟可加速
    device: str = "auto"  # auto | cuda | cpu


@dataclass
class CheckpointSection:
    """Checkpoint 與續訓設定。"""

    save_every_steps: int = 500
    keep_last: int = 3
    resume: str = "auto"  # auto（自動找最新）| none（從頭訓練）| 指定 checkpoint 路徑
    save_best: bool = True
    monitor: str = "val/mAP50"
    monitor_mode: str = "max"  # max | min
    local_tmp_dir: Optional[str] = None  # 先寫到本機暫存再複製到 Drive；預設為系統暫存目錄


@dataclass
class LoggingSection:
    num_vis_samples: int = 4  # 每次評估寫進 TensorBoard 的預測視覺化張數


@dataclass
class ExperimentConfig:
    """完整的實驗設定，對應單一 YAML 檔。"""

    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    data: DataSection = field(default_factory=DataSection)
    model: ModelSection = field(default_factory=ModelSection)
    optim: OptimSection = field(default_factory=OptimSection)
    scheduler: SchedulerSection = field(default_factory=SchedulerSection)
    train: TrainSection = field(default_factory=TrainSection)
    checkpoint: CheckpointSection = field(default_factory=CheckpointSection)
    logging: LoggingSection = field(default_factory=LoggingSection)

    # ---- 衍生路徑 ----
    @property
    def run_dir(self) -> Path:
        return Path(self.experiment.output_root) / self.experiment.name

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"


# --------------------------------------------------------------------------- #
# 載入與輸出
# --------------------------------------------------------------------------- #
def load_config(path: str | Path, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    """載入 YAML 並套用命令列覆寫（格式：a.b.c=value）。

    命令列覆寫是「整個取代」而不是合併：例如 model.backbone.kwargs={} 會清空 YAML 裡的 kwargs，
    而不是保留原本的 key。
    """
    merged = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.load(str(path)))
    for item in overrides or []:
        key = item.split("=", 1)[0]
        node = OmegaConf.select(merged, key, default=_MISSING)
        if node is _MISSING and not _is_free_dict(OmegaConf.select(merged, key.rpartition(".")[0])):
            raise KeyError(f"設定中沒有 {key} 這個欄位，請檢查拼字")
        if _is_free_dict(node):
            # kwargs 這類自由格式的 dict：先清空，覆寫後才不會殘留 YAML 裡原本的 key
            OmegaConf.update(merged, key, {}, merge=False)
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([item]))
    cfg: ExperimentConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    _validate(cfg)
    return cfg


def _is_free_dict(node: Any) -> bool:
    """是否為 kwargs 這類 Dict[str, Any] 欄位（可自由新增 key，不是固定 schema）。"""
    return isinstance(node, DictConfig) and OmegaConf.get_type(node) is dict


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """轉成純 dict，方便存進 checkpoint。"""
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]


def config_to_yaml(cfg: ExperimentConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))


def _validate(cfg: ExperimentConfig) -> None:
    """檢查 OmegaConf 型別系統無法表達的限制。"""
    if cfg.experiment.task not in {"detection", "semantic_seg"}:
        raise ValueError(f"未知的 task：{cfg.experiment.task}")
    if cfg.train.max_steps is None and cfg.train.max_epochs is None:
        raise ValueError("train.max_steps 與 train.max_epochs 至少要設定一個")
    if cfg.train.precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"未知的 precision：{cfg.train.precision}")
    if cfg.checkpoint.monitor_mode not in {"max", "min"}:
        raise ValueError("checkpoint.monitor_mode 只能是 max 或 min")
    if not cfg.data.train_sets or not cfg.data.val_sets:
        raise ValueError("data.train_sets 與 data.val_sets 不可為空")
    if cfg.train.grad_accum_steps < 1:
        raise ValueError("train.grad_accum_steps 必須 >= 1")
