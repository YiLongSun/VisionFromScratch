"""Checkpoint 管理：安全寫入、保留最近 N 個、自動找最新且可讀的檔案。

寫入流程（避免 Colab 斷線時留下損壞的檔案）：
    1. 先 torch.save 到本機暫存目錄（Drive 的 FUSE 對大量隨機寫入很慢）
    2. 複製到 Drive 上的 *.tmp
    3. os.replace 改名成正式檔名（改名是原子操作，正式檔名永遠是完整的檔案）
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..utils.logger import get_logger

logger = get_logger(__name__)

_STEP_PATTERN = re.compile(r"^step_(\d+)\.pt$")
BEST_NAME = "best.pt"


class CheckpointManager:
    def __init__(self, directory: str | Path, keep_last: int = 3, local_tmp_dir: Optional[str] = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, keep_last)
        self.local_tmp_dir = Path(local_tmp_dir or tempfile.gettempdir())
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- 寫入 ----
    def save(self, state: Dict[str, Any], step: int) -> Path:
        path = self._atomic_save(state, f"step_{step:08d}.pt")
        self._prune()
        logger.info("已儲存 checkpoint：%s", path)
        return path

    def save_best(self, state: Dict[str, Any]) -> Path:
        path = self._atomic_save(state, BEST_NAME)
        logger.info("已更新最佳模型：%s", path)
        return path

    def _atomic_save(self, state: Dict[str, Any], name: str) -> Path:
        final_path = self.directory / name
        local_path = self.local_tmp_dir / f"vfs_{os.getpid()}_{name}"
        remote_tmp = self.directory / f"{name}.tmp"
        try:
            torch.save(state, local_path)
            shutil.copyfile(local_path, remote_tmp)
            os.replace(remote_tmp, final_path)
        finally:
            local_path.unlink(missing_ok=True)
            remote_tmp.unlink(missing_ok=True)
        return final_path

    def _prune(self) -> None:
        for _, path in self.list_checkpoints()[: -self.keep_last]:
            path.unlink(missing_ok=True)

    # ---- 讀取 ----
    def list_checkpoints(self) -> List[Tuple[int, Path]]:
        """依 step 由小到大排序的 (step, 路徑)。"""
        items = []
        for path in self.directory.iterdir():
            match = _STEP_PATTERN.match(path.name)
            if match:
                items.append((int(match.group(1)), path))
        return sorted(items)

    def load_latest(self, map_location: str | torch.device = "cpu") -> Optional[Dict[str, Any]]:
        """從最新的開始嘗試讀取；檔案損壞就退回前一個。全部都沒有則回傳 None。"""
        for step, path in reversed(self.list_checkpoints()):
            try:
                state = load_checkpoint(path, map_location)
                logger.info("找到 checkpoint：%s（step %d）", path, step)
                return state
            except Exception as error:  # noqa: BLE001 - 任何讀取錯誤都視為損壞
                logger.warning("checkpoint %s 無法讀取（%s），改用前一個", path, error)
        return None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """讀取本專案存的 checkpoint。

    RNG 狀態包含 NumPy 陣列與 Python tuple，必須 weights_only=False；
    請只載入自己產生或信任的檔案。
    """
    return torch.load(path, map_location=map_location, weights_only=False)
