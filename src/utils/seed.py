"""亂數種子與 RNG 狀態的儲存／還原。"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """設定 Python、NumPy、PyTorch（CPU 與所有 GPU）的亂數種子。"""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
    """擷取目前所有 RNG 狀態，存進 checkpoint 用。"""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """還原 RNG 狀態；若 GPU 數量不同則只還原能對應的部分。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda")
    if cuda_states and torch.cuda.is_available():
        n = min(len(cuda_states), torch.cuda.device_count())
        for i in range(n):
            torch.cuda.set_rng_state(cuda_states[i], device=i)
