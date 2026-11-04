"""精簡的訓練迴圈，重點是「隨時可中斷、精確續訓」。

Checkpoint 內容：
    model / optimizer / scheduler / scaler 的 state_dict
    TrainState：已完成的 step、目前 epoch、本 epoch 已消耗的樣本數、最佳指標
    RNG：Python、NumPy、PyTorch CPU、CUDA
    config：當時的完整設定（方便日後查閱）

續訓流程：
    1. 啟動時在 checkpoint 目錄找最新且可讀的檔案（resume: auto）
    2. 還原所有狀態，sampler 依 (seed, epoch) 重建同樣的順序並跳過已訓練的樣本
    3. TensorBoard 以 purge_step 清掉 checkpoint 之後、斷線之前寫下的紀錄，曲線不會重疊
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..config import ExperimentConfig, config_to_dict, config_to_yaml
from ..data import build_train_loader, build_val_loader
from ..data.loaders import num_batches_for
from ..tasks import Task
from ..utils.logger import get_logger
from ..utils.misc import count_parameters, format_seconds, resolve_device
from ..utils.seed import get_rng_state, seed_everything, set_rng_state
from .checkpoint import CheckpointManager, load_checkpoint
from .optim import build_optimizer, build_scheduler

logger = get_logger(__name__)


@dataclass
class TrainState:
    step: int = 0  # 已完成的 optimizer step 數
    epoch: int = 0
    samples_in_epoch: int = 0  # 本 epoch 已送進模型的樣本數（續訓時要跳過）
    best_metric: Optional[float] = None
    last_saved_step: int = -1


class Trainer:
    def __init__(self, cfg: ExperimentConfig, task: Task) -> None:
        self.cfg = cfg
        self.task = task
        self.device = resolve_device(cfg.train.device)
        torch.backends.cudnn.benchmark = cfg.train.cudnn_benchmark
        seed_everything(cfg.experiment.seed)

        # ---- 資料 ----
        self.train_set, self.val_set = task.build_datasets()
        self.train_loader, self.sampler = build_train_loader(
            self.train_set,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            collate_fn=task.collate_fn,
            seed=cfg.experiment.seed,
            pin_memory=cfg.data.pin_memory,
        )
        self.val_loader = build_val_loader(
            self.val_set,
            batch_size=cfg.data.val_batch_size,
            num_workers=cfg.data.num_workers,
            collate_fn=task.collate_fn,
            pin_memory=cfg.data.pin_memory,
        )
        self.accum = cfg.train.grad_accum_steps
        batches_per_epoch = len(self.train_set) // cfg.data.batch_size
        self.steps_per_epoch = max(1, batches_per_epoch // self.accum)
        self.total_steps = cfg.train.max_steps or cfg.train.max_epochs * self.steps_per_epoch  # type: ignore[operator]

        # ---- 模型與最佳化 ----
        self.model = task.build_model().to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.scheduler, self.total_steps)
        self.amp_dtype = self._resolve_precision(cfg.train.precision)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)

        # ---- 輸出與續訓 ----
        self.monitor = cfg.checkpoint.monitor or task.default_monitor
        self.ckpt = CheckpointManager(cfg.checkpoint_dir, cfg.checkpoint.keep_last, cfg.checkpoint.local_tmp_dir)
        self.state = TrainState()
        self._resume()
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        (cfg.run_dir / "config.yaml").write_text(config_to_yaml(cfg), encoding="utf-8")
        self.writer = SummaryWriter(str(cfg.tensorboard_dir), purge_step=self.state.step or None)

        total, trainable = count_parameters(self.model)
        logger.info(
            "訓練集 %d 張、驗證集 %d 張；每 epoch %d step，總共 %d step；參數 %.1fM（可訓練 %.1fM）",
            len(self.train_set),
            len(self.val_set),
            self.steps_per_epoch,
            self.total_steps,
            total / 1e6,
            trainable / 1e6,
        )

    # ------------------------------------------------------------------ #
    # 混合精度
    # ------------------------------------------------------------------ #
    def _resolve_precision(self, precision: str) -> Optional[torch.dtype]:
        if precision == "fp32":
            return None
        if self.device.type != "cuda":
            logger.warning("非 CUDA 裝置不使用混合精度，改用 fp32")
            return None
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("此 GPU 不支援 bf16（例如 T4），改用 fp16")
            return torch.float16
        return torch.bfloat16 if precision == "bf16" else torch.float16

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def _state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "train_state": asdict(self.state),
            "rng": get_rng_state(),
            "config": config_to_dict(self.cfg),
        }

    def _save(self) -> None:
        if self.state.last_saved_step == self.state.step:
            return
        self.state.last_saved_step = self.state.step
        self.ckpt.save(self._state_dict(), self.state.step)

    def _resume(self) -> None:
        mode = self.cfg.checkpoint.resume
        if mode == "none":
            logger.info("resume=none：從頭開始訓練")
            return
        checkpoint = self.ckpt.load_latest() if mode == "auto" else load_checkpoint(mode)
        if checkpoint is None:
            logger.info("沒有找到 checkpoint，從頭開始訓練")
            return
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.state = TrainState(**checkpoint["train_state"])
        set_rng_state(checkpoint["rng"])
        if checkpoint.get("config") != config_to_dict(self.cfg):
            logger.warning("目前的設定與 checkpoint 存檔時不同，請確認這是你要的（例如只改了 eval 頻率）")
        logger.info(
            "從 step %d 續訓（epoch %d，本 epoch 已訓練 %d 張）",
            self.state.step,
            self.state.epoch,
            self.state.samples_in_epoch,
        )

    # ------------------------------------------------------------------ #
    # 主迴圈
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        try:
            if self.state.step < self.total_steps:
                self._train_loop()
            self._save()
            self._run_eval(full=True)
        except KeyboardInterrupt:
            logger.warning("收到中斷訊號，儲存 checkpoint 後結束")
            self._save()
        finally:
            self.writer.close()

    def _train_loop(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._meters: Dict[str, float] = defaultdict(float)
        self._meter_count = 0
        self._timer = time.time()
        self._timer_step = self.state.step
        batch_size = self.cfg.data.batch_size

        while self.state.step < self.total_steps:
            self.sampler.set_epoch(self.state.epoch, start_index=self.state.samples_in_epoch)
            micro_step = 0
            for batch in self.train_loader:
                self._backward(batch)
                micro_step += 1
                if micro_step % self.accum != 0:
                    continue
                self._optimizer_step()
                # 只在 optimizer step 完成時推進進度，中斷時未完成的梯度累積會在續訓時重跑
                self.state.samples_in_epoch += batch_size * self.accum
                self.state.step += 1
                self._after_step()
                if self.state.step >= self.total_steps:
                    return
            # epoch 結束：丟掉不足一次累積的梯度，進入下一個 epoch
            self.optimizer.zero_grad(set_to_none=True)
            self.state.epoch += 1
            self.state.samples_in_epoch = 0

    def _backward(self, batch: Any) -> None:
        with self.autocast():
            losses = self.task.training_step(self.model, batch, self.device)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            logger.warning("step %d 出現非有限的 loss（%s），略過此 batch", self.state.step, loss.item())
            return
        self.scaler.scale(loss / self.accum).backward()
        for name, value in losses.items():
            self._meters[name] += float(value.detach())
        self._meter_count += 1

    def _optimizer_step(self) -> None:
        clip = self.cfg.train.grad_clip_norm
        if clip is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self._meters["grad_norm"] += float(grad_norm) * self.accum  # 之後會除以 micro-batch 數
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def _after_step(self) -> None:
        step = self.state.step
        if step % self.cfg.train.log_every_steps == 0:
            self._log_train()
        if step % self.cfg.checkpoint.save_every_steps == 0:
            self._save()
        if step % self.cfg.train.eval_every_steps == 0 and step < self.total_steps:
            self._run_eval(full=False)
            self.model.train()

    # ------------------------------------------------------------------ #
    # Log 與評估
    # ------------------------------------------------------------------ #
    def _log_train(self) -> None:
        step = self.state.step
        count = max(self._meter_count, 1)
        averages = {name: total / count for name, total in self._meters.items()}
        for name, value in averages.items():
            self.writer.add_scalar(f"train/{name}", value, step)
        for group in self.optimizer.param_groups:
            self.writer.add_scalar(f"lr/{group.get('name', 'group')}", group["lr"], step)
        if self.amp_dtype == torch.float16:
            self.writer.add_scalar("train/amp_scale", self.scaler.get_scale(), step)

        elapsed = time.time() - self._timer
        steps_done = step - self._timer_step
        speed = steps_done / max(elapsed, 1e-6)
        eta = (self.total_steps - step) / max(speed, 1e-6)
        self.writer.add_scalar("perf/steps_per_sec", speed, step)
        memory = ""
        if self.device.type == "cuda":
            gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
            self.writer.add_scalar("perf/max_memory_gb", gb, step)
            memory = f" | mem {gb:.1f}GB"
        logger.info(
            "step %d/%d | epoch %d | loss %.4f | lr %.2e | %.2f step/s | ETA %s%s",
            step,
            self.total_steps,
            self.state.epoch,
            averages.get("loss", math.nan),
            self.optimizer.param_groups[0]["lr"],
            speed,
            format_seconds(eta),
            memory,
        )
        self._meters.clear()
        self._meter_count = 0
        self._timer = time.time()
        self._timer_step = step

    def _run_eval(self, full: bool) -> Dict[str, float]:
        step = self.state.step
        max_batches = None if full else num_batches_for(self.cfg.data.val_max_samples, self.cfg.data.val_batch_size)
        logger.info("開始評估（%s）…", "完整驗證集" if max_batches is None else f"前 {max_batches} 個 batch")
        start = time.time()
        metrics = self.task.evaluate(self.model, self.val_loader, self.device, self.autocast, max_batches)
        prefix = "val" if full or max_batches is None else "val_partial"
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, step)
        images = self.task.visualize(
            self.model, self.val_set, self.device, self.autocast, self.cfg.logging.num_vis_samples
        )
        for tag, image in images.items():
            self.writer.add_image(tag, image, step)
        self.writer.flush()

        headline = {k: round(v, 4) for k, v in metrics.items() if "/" not in k}
        logger.info("評估完成（%.0f 秒）：%s", time.time() - start, headline)
        # 只拿「同一組評估資料」的結果互相比較：有設定 val_max_samples 時，以定期評估的子集決定 best
        periodic_prefix = "val" if self.cfg.data.val_max_samples is None else "val_partial"
        if prefix == periodic_prefix:
            self._update_best(metrics, prefix)
        if full:
            path = self.cfg.run_dir / "final_metrics.json"
            path.write_text(json.dumps({"step": step, **metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
        return metrics

    def _update_best(self, metrics: Dict[str, float], prefix: str) -> None:
        """以 monitor 指標決定是否更新 best.pt（部分驗證集與完整驗證集都會比較）。"""
        if not self.cfg.checkpoint.save_best:
            return
        name = self.monitor.split("/", 1)[-1]
        if name not in metrics or math.isnan(metrics[name]):
            return
        value = metrics[name]
        best = self.state.best_metric
        better = best is None or (value > best if self.cfg.checkpoint.monitor_mode == "max" else value < best)
        if better:
            self.state.best_metric = value
            self.ckpt.save_best(
                {
                    "model": self.model.state_dict(),
                    "step": self.state.step,
                    "metrics": metrics,
                    "metric_source": prefix,
                    "config": config_to_dict(self.cfg),
                }
            )
