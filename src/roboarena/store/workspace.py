"""工作区：一个目录，放数据库和全部运行产物。离线命令与服务只通过它交换数据。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType

from roboarena.store.db import connect

SUBDIRS = ("runs", "artifacts", "suites", "datasets", "reports", "cache")


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        for name in SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.db: sqlite3.Connection = connect(self.root / "roboarena.db")

    def rel(self, path: Path) -> str:
        """数据库里只存相对工作区的路径，工作区整体搬家后仍然可用。"""
        return path.resolve().relative_to(self.root).as_posix()

    def abs(self, rel: str) -> Path:
        return self.root / rel

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Workspace:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
