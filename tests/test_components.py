"""各元件的單元測試：評估指標、sampler、scheduler、外部權重載入、checkpoint。"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import timm
import torch

from src.config import BackboneSection, SchedulerSection, load_config
from src.data import ResumableRandomSampler, VOCDetectionDataset, VOCSegmentationDataset
from src.engine import CheckpointManager, build_scheduler
from src.metrics import SegmentationEvaluator, VOCDetectionEvaluator
from src.models.backbone import TimmBackbone


# ---------------------------------------------------------------- 評估指標 --
def _gt(boxes, labels, difficult=None):
    n = len(labels)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels),
        "difficult": torch.tensor(difficult or [False] * n),
    }


def _pred(boxes, labels, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels),
        "scores": torch.tensor(scores),
    }


def test_voc_map_perfect_predictions():
    ev = VOCDetectionEvaluator(["a", "b"])
    ev.update(
        [_pred([[0, 0, 10, 10], [5, 5, 20, 20]], [1, 2], [0.9, 0.8])], [_gt([[0, 0, 10, 10], [5, 5, 20, 20]], [1, 2])]
    )
    m = ev.compute()
    assert m["mAP50"] == pytest.approx(1.0)
    assert m["mAP50_95"] == pytest.approx(1.0)


def test_voc_map_duplicate_is_false_positive():
    ev = VOCDetectionEvaluator(["a"])
    # 同一個 GT 被偵測兩次：第一個是 TP、第二個是 FP，FP 出現在 recall 已達 1 之後，AP 仍為 1
    ev.update([_pred([[0, 0, 10, 10], [0, 0, 10, 10]], [1, 1], [0.9, 0.8])], [_gt([[0, 0, 10, 10]], [1])])
    assert ev.compute()["mAP50"] == pytest.approx(1.0)
    # 分數最高的是錯誤框：recall 達 1 時 precision 只有 1/2
    ev = VOCDetectionEvaluator(["a"])
    ev.update([_pred([[50, 50, 60, 60], [0, 0, 10, 10]], [1, 1], [0.9, 0.8])], [_gt([[0, 0, 10, 10]], [1])])
    assert ev.compute()["mAP50"] == pytest.approx(0.5)


def test_voc_map_difficult_is_ignored():
    ev = VOCDetectionEvaluator(["a"])
    gt = _gt([[0, 0, 10, 10], [30, 30, 40, 40]], [1, 1], [False, True])
    ev.update([_pred([[30, 30, 40, 40], [0, 0, 10, 10]], [1, 1], [0.9, 0.8])], [gt])
    assert ev.compute()["mAP50"] == pytest.approx(1.0)


def test_miou():
    ev = SegmentationEvaluator(["bg", "fg"])
    target = torch.tensor([[[0, 0, 1, 255]]])
    pred = torch.tensor([[[0, 1, 1, 0]]])
    ev.update(pred, target)
    m = ev.compute()
    # bg: TP 1, FN 1, FP 0 → 1/2；fg: TP 1, FP 1 → 1/2
    assert m["mIoU"] == pytest.approx(0.5)


# ---------------------------------------------------------------- sampler --
def test_sampler_resume_matches_uninterrupted():
    sampler = ResumableRandomSampler(20, seed=3)
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_epoch(2, start_index=7)
    assert list(sampler) == full[7:]
    sampler.set_epoch(3)
    assert [i for i, _ in sampler] != [i for i, _ in full]  # 不同 epoch 順序不同
    assert sorted(i for i, _ in full) == list(range(20))


# -------------------------------------------------------------- scheduler --
def test_scheduler_warmup_and_cosine():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = build_scheduler(opt, SchedulerSection(name="cosine", warmup_steps=10, warmup_start_factor=0.0), 110)
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[10] == pytest.approx(1.0)
    assert lrs[60] == pytest.approx(0.5, abs=1e-6)
    assert lrs[-1] < 0.01


def test_scheduler_multistep():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = build_scheduler(opt, SchedulerSection(name="multistep", warmup_steps=0, milestones=[0.5]), 100)
    for _ in range(50):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


# ---------------------------------------------------------- 外部權重載入 --
def test_external_weights_with_prefix(tmp_path):
    """模擬 DINO ResNet-50：torchvision 命名、帶 module. 前綴、還有用不到的 fc。"""
    source = timm.create_model("resnet18", pretrained=False)
    state = {f"module.{k}": v for k, v in source.state_dict().items()}
    path = tmp_path / "weights.pth"
    torch.save({"teacher": state}, path)

    cfg = BackboneSection(
        name="resnet18", pretrained=False, checkpoint=str(path), checkpoint_key="teacher", freeze_bn=False
    )
    backbone = TimmBackbone(cfg)
    assert torch.equal(backbone.body.conv1.weight, source.conv1.weight)


# ------------------------------------------------------------- checkpoint --
def test_checkpoint_keep_last_and_corrupt_fallback(tmp_path):
    manager = CheckpointManager(tmp_path, keep_last=2)
    for step in (10, 20, 30):
        manager.save({"step": step}, step)
    assert [s for s, _ in manager.list_checkpoints()] == [20, 30]
    (tmp_path / "step_00000030.pt").write_bytes(b"broken")
    assert manager.load_latest()["step"] == 20


# --------------------------------------------------------------- dataset --
def test_datasets(fake_voc):
    det = VOCDetectionDataset(fake_voc, [("2007", "trainval"), ("2012", "trainval")], use_difficult=False)
    assert len(det) == 24
    image, target = det[0]
    assert image.dtype == torch.uint8 and target["boxes"].shape[-1] == 4
    assert not target["difficult"].any()

    seg = VOCSegmentationDataset(fake_voc, [("2012", "train")])
    image, mask = seg[0]
    assert mask.shape == image.shape[-2:]
    assert set(mask.unique().tolist()) <= set(range(21)) | {255}
    assert math.isfinite(float(image.float().mean()))


# ------------------------------------------------------------------- 設定 --
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_config_override_replaces_free_dict():
    cfg = load_config(CONFIG_DIR / "segmentation.yaml", ["model.backbone.kwargs={}"])
    assert cfg.model.backbone.kwargs == {}  # 整個取代，不殘留 output_stride
    cfg = load_config(CONFIG_DIR / "detection.yaml", ["model.head.kwargs.nms_thresh=0.5"])
    assert cfg.model.head.kwargs["nms_thresh"] == 0.5 and "score_thresh" in cfg.model.head.kwargs


def test_config_override_rejects_typo():
    with pytest.raises(KeyError):
        load_config(CONFIG_DIR / "detection.yaml", ["optim.lrr=0.1"])
