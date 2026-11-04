"""訓練入口。

    python -m src.train --config configs/detection.yaml
    python -m src.train --config configs/detection.yaml optim.lr=0.005 train.max_epochs=6

若 checkpoint 目錄已有存檔，會自動從最新的 step 接著訓練（checkpoint.resume=auto）。
"""

from __future__ import annotations

import argparse

from .config import config_to_yaml, load_config
from .engine import Trainer
from .tasks import build_task
from .utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionFromScratch 訓練")
    parser.add_argument("--config", required=True, help="YAML 設定檔路徑")
    parser.add_argument("overrides", nargs="*", help="覆寫設定，格式 a.b.c=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger = get_logger()
    logger.info("實驗設定：\n%s", config_to_yaml(cfg))
    Trainer(cfg, build_task(cfg)).fit()


if __name__ == "__main__":
    main()
