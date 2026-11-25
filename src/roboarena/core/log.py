"""结构化日志：每行一个 JSON 对象，带追踪编号。

用法：
    log = get_logger(__name__)
    log.info("评测完成", extra={"policy": "bc-50", "n": 50})
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

ROOT = "roboarena"

# 当前请求或任务的追踪编号。服务的中间件和任务执行器负责设置它（M5a）。
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (trace_id := trace_id_var.get()) is not None:
            entry["trace_id"] = trace_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith(ROOT) else f"{ROOT}.{name}")


def setup_logging(level: str = "INFO", stream: IO[str] | None = None) -> None:
    """配置 roboarena 根日志器，输出到 stream（默认标准错误）。重复调用不会叠加处理器。"""
    root = logging.getLogger(ROOT)
    for handler in list(root.handlers):
        if not isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def add_log_file(path: Path) -> logging.Handler:
    """把日志同时写进一个文件（运行目录里的 log.txt）。用完后交给 remove_log_file。"""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logging.getLogger(ROOT).addHandler(handler)
    return handler


def remove_log_file(handler: logging.Handler) -> None:
    logging.getLogger(ROOT).removeHandler(handler)
    handler.close()
