"""簡單的註冊表：讓 YAML 裡的字串對應到建構函式。

用法：
    NECKS = Registry("neck")

    @NECKS.register("fpn")
    class FPN(nn.Module): ...

    neck = NECKS.build("fpn", in_channels=[...])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Callable[..., Any]] = {}

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            key = name or obj.__name__
            if key in self._entries:
                raise KeyError(f"{self.kind} '{key}' 已經註冊過了")
            self._entries[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._entries:
            raise KeyError(f"未知的 {self.kind}：'{name}'，可用的有 {self.names()}")
        return self._entries[name]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
