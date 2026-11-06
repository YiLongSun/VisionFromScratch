"""可從 epoch 中途精確接續的 sampler。

一般的 RandomSampler 每次建立 iterator 都會重新洗牌，續訓時無法得知「這個 epoch 已經看過哪些樣本」。
這裡改成：
1. 每個 epoch 的順序只由 (seed, epoch) 決定，續訓時可以重建出一模一樣的順序。
2. 可指定從第幾個樣本開始，跳過中斷前已經訓練過的部分。
3. 每個樣本另外配一個專屬的亂數種子，資料增強的隨機性因此與 worker 數量、是否中斷都無關。
"""

from __future__ import annotations

import random
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

_MAX_SEED = 2**31 - 1


class ResumableRandomSampler(Sampler[Tuple[int, int]]):
    """產生 (樣本索引, 樣本種子) 的 sampler，需搭配 SeededDataset 使用。"""

    def __init__(self, num_samples: int, seed: int, shuffle: bool = True) -> None:
        self.num_samples = num_samples
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        """每個 epoch 開始前呼叫；start_index 為本 epoch 要跳過的樣本數。"""
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(f"start_index={start_index} 超出範圍 [0, {self.num_samples}]")
        self.epoch = epoch
        self.start_index = start_index

    def _epoch_plan(self) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=generator)
        else:
            order = torch.arange(self.num_samples)
        seeds = torch.randint(0, _MAX_SEED, (self.num_samples,), generator=generator)
        return order, seeds

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        order, seeds = self._epoch_plan()
        for pos in range(self.start_index, self.num_samples):
            yield int(order[pos]), int(seeds[pos])

    def __len__(self) -> int:
        return self.num_samples - self.start_index


class SeededDataset(Dataset):
    """包裝任意 Dataset：取樣本前先用 sampler 給的種子設定 RNG，取完再還原。

    還原 RNG 是為了不影響主程序（num_workers=0 時資料在主程序讀取）裡的模型隨機性，例如 dropout。
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self.dataset[key]
        index, seed = key
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
