"""端對端測試：每種模型組合都能訓練與評估，以及「中斷後續訓」與「不中斷」的結果完全相同。"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch

from src.config import load_config
from src.engine import Trainer
from src.tasks import build_task

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _tiny_overrides(root: Path, out: Path) -> List[str]:
    """把設定縮到能在 CPU 上幾秒內跑完。"""
    return [
        f"data.root={root}",
        f"experiment.output_root={out}",
        "data.batch_size=2",
        "data.val_batch_size=2",
        "data.num_workers=0",
        "data.augment.min_size=[96]",
        "data.augment.max_size=160",
        "data.augment.test_min_size=96",
        "data.augment.base_size=96",
        "data.augment.crop_size=64",
        "train.max_steps=4",
        "train.max_epochs=null",
        "train.log_every_steps=2",
        "train.eval_every_steps=2",
        "train.device=cpu",
        "checkpoint.save_every_steps=2",
        "scheduler.warmup_steps=2",
        "model.backbone.pretrained=false",
        "model.backbone.checkpoint=null",
        "logging.num_vis_samples=1",
    ]


# 對應 configs/*.yaml 檔尾的「常見組合」，backbone 換成小模型以加快測試
CASES = [
    ("detection.yaml", ["model.backbone.name=resnet18"]),
    (
        "detection.yaml",
        [
            "model.backbone.name=resnet18",
            "model.backbone.out_indices=[1,2,3,4]",
            "model.neck.extra_levels=maxpool",
            "model.head.name=faster_rcnn",
            "model.head.kwargs={box_score_thresh: 0.05}",
        ],
    ),
    (
        "detection.yaml",
        [
            "model.backbone.name=resnet18",
            "model.head.name=retinanet",
            "model.head.kwargs={score_thresh: 0.05, detections_per_img: 100}",
        ],
    ),
    ("detection.yaml", ["model.backbone.name=convnext_atto", "model.backbone.out_indices=[1,2,3]"]),
    (
        "detection.yaml",
        [
            "model.backbone.name=vit_tiny_patch16_224",
            "model.backbone.out_indices=[11]",
            "model.neck.name=simple_pyramid",
            "model.neck.scales=[2.0,1.0,0.5]",
        ],
    ),
    ("segmentation.yaml", ["model.backbone.name=resnet18"]),
    (
        "segmentation.yaml",
        [
            "model.backbone.name=resnet18",
            "model.backbone.out_indices=[1,2,3,4]",
            "model.backbone.kwargs={}",
            "model.neck.name=fpn",
            "model.head.name=fpn_seg",
            "model.head.kwargs={channels: 64}",
        ],
    ),
    (
        "segmentation.yaml",
        [
            "model.backbone.name=vit_small_patch14_dinov2",
            "model.backbone.out_indices=[10,11]",
            "model.backbone.freeze=true",
            "model.backbone.kwargs={}",
            "model.head.name=linear",
            "model.head.kwargs={}",
        ],
    ),
    (
        "segmentation.yaml",
        [
            "model.backbone.name=vit_tiny_patch16_224",
            "model.backbone.out_indices=[11]",
            "model.backbone.kwargs={}",
            "model.neck.name=simple_pyramid",
            "model.head.name=fpn_seg",
            "model.head.kwargs={}",
        ],
    ),
]


@pytest.mark.parametrize("config_name,extra", CASES)
def test_train_and_evaluate(fake_voc, tmp_path, config_name, extra):
    cfg = load_config(CONFIG_DIR / config_name, _tiny_overrides(fake_voc, tmp_path) + extra)
    Trainer(cfg, build_task(cfg)).fit()
    assert (cfg.checkpoint_dir / "step_00000004.pt").is_file()
    assert (cfg.run_dir / "final_metrics.json").is_file()
    assert any(cfg.tensorboard_dir.iterdir())


def _final_weights(cfg) -> dict:
    return torch.load(cfg.checkpoint_dir / "step_00000006.pt", weights_only=False)["model"]


@pytest.mark.parametrize(
    "config_name,extra",
    [
        ("detection.yaml", ["model.backbone.name=resnet18"]),
        ("segmentation.yaml", ["model.backbone.name=resnet18", "train.grad_accum_steps=2"]),
    ],
)
def test_resume_is_exact(fake_voc, tmp_path, monkeypatch, config_name, extra):
    """跑 6 step 不中斷 vs 在第 3 step 按下中斷再續訓，最後的權重必須逐位元相同。

    每個 epoch 有 12/2=6 個 batch，第 3 step 中斷時剛好在 epoch 中間（含梯度累積時則跨 epoch）。
    """
    common = (
        _tiny_overrides(fake_voc, tmp_path)
        + extra
        + [
            "train.max_steps=6",
            "train.eval_every_steps=100",
            "checkpoint.save_every_steps=100",
        ]
    )
    cfg_a = load_config(CONFIG_DIR / config_name, common + ["experiment.name=uninterrupted"])
    Trainer(cfg_a, build_task(cfg_a)).fit()

    cfg_b = load_config(CONFIG_DIR / config_name, common + ["experiment.name=interrupted"])
    original_after_step = Trainer._after_step

    def interrupt_at_3(self):
        original_after_step(self)
        if self.state.step == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(Trainer, "_after_step", interrupt_at_3)
    Trainer(cfg_b, build_task(cfg_b)).fit()
    assert (cfg_b.checkpoint_dir / "step_00000003.pt").is_file()

    monkeypatch.setattr(Trainer, "_after_step", original_after_step)
    Trainer(cfg_b, build_task(cfg_b)).fit()

    weights_a, weights_b = _final_weights(cfg_a), _final_weights(cfg_b)
    for name, tensor in weights_a.items():
        assert torch.equal(tensor, weights_b[name]), f"{name} 不一致"
