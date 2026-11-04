"""統一的 logging 設定。"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "vfs") -> logging.Logger:
    """取得 logger；第一次呼叫時設定輸出到 stdout（Colab cell 才看得到）。"""
    root = logging.getLogger("vfs")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    if name == "vfs":
        return root
    return root.getChild(name.removeprefix("src."))
