"""評估入口：載入 checkpoint，在完整驗證集上計算指標。

python -m src.evaluate --config configs/detection.yaml              # 預設用 best.pt
python -m src.evaluate --config configs/detection.yaml --checkpoint latest
python -m src.evaluate --config configs/detection.yaml --checkpoint /path/to/step_00010000.pt
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from .config import load_config
from .data import build_val_loader
from .engine import CheckpointManager, load_checkpoint
from .engine.checkpoint import BEST_NAME
from .tasks import build_task
from .utils.logger import get_logger
from .utils.misc import resolve_device

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionFromScratch 評估")
    parser.add_argument("--config", required=True, help="YAML 設定檔路徑")
    parser.add_argument("--checkpoint", default="best", help="best | latest | checkpoint 路徑")
    parser.add_argument("overrides", nargs="*", help="覆寫設定，格式 a.b.c=value")
    return parser.parse_args()


def _resolve_checkpoint(cfg, choice: str) -> Path:
    if choice == "best":
        return cfg.checkpoint_dir / BEST_NAME
    if choice == "latest":
        items = CheckpointManager(cfg.checkpoint_dir).list_checkpoints()
        if not items:
            raise FileNotFoundError(f"{cfg.checkpoint_dir} 內沒有 checkpoint")
        return items[-1][1]
    return Path(choice)


def main() -> None:
    args = parse_args()
    # 評估時不需要下載預訓練權重，checkpoint 會覆蓋所有參數
    overrides = ["model.backbone.pretrained=false", "model.backbone.checkpoint=null", *args.overrides]
    cfg = load_config(args.config, overrides)
    device = resolve_device(cfg.train.device)
    task = build_task(cfg)

    path = _resolve_checkpoint(cfg, args.checkpoint)
    model = task.build_model().to(device)
    model.load_state_dict(load_checkpoint(path)["model"])
    logger.info("已載入 %s", path)

    _, val_set = task.build_datasets()
    loader = build_val_loader(val_set, cfg.data.val_batch_size, cfg.data.num_workers, task.collate_fn)
    use_amp = device.type == "cuda" and cfg.train.precision != "fp32"

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    metrics = task.evaluate(model, loader, device, autocast)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
