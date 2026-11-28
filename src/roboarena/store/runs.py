"""运行记录：每次训练或评测一个目录、一行数据库记录。

runs/<run_id>/
    config.toml     生效的完整配置
    meta.json       代码版本、是否有未提交改动、数据指纹、设备、依赖锁文件摘要、起止时间
    metrics.jsonl   每行一条：步数、指标名、值
    log.txt         本次运行的结构化日志
"""

from __future__ import annotations

import json
import platform
import secrets
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from roboarena.core.errors import ConflictError, NotFoundError
from roboarena.core.hashing import canonical_json, sha256_file, sha256_text
from roboarena.core.log import add_log_file, remove_log_file
from roboarena.store.db import now
from roboarena.store.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_info(cwd: Path = REPO_ROOT) -> tuple[str | None, bool | None]:
    """返回（提交号，工作区是否有未提交改动）。不在 git 仓库里时两者都是 None。"""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return rev, bool(status.strip())


@dataclass
class Run:
    id: str
    kind: str
    dir: Path
    ws: Workspace

    def log_metric(self, step: int, name: str, value: float) -> None:
        with open(self.dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "name": name, "value": float(value)}) + "\n")

    def update_meta(self, **fields: Any) -> None:
        path = self.dir / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(fields)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _finish(self, status: str) -> None:
        finished = now()
        self.update_meta(status=status, finished_at=finished)
        self.ws.db.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?", (status, finished, self.id)
        )
        self.ws.db.commit()


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@contextmanager
def start_run(
    ws: Workspace,
    kind: str,
    *,
    config: Mapping[str, Any],
    config_toml: str,
    seed: int,
    dataset_fp: str | None = None,
    device: str | None = None,
) -> Iterator[Run]:
    """开始一次运行。正常退出记为 done，抛异常记为 failed，Ctrl-C 记为 interrupted。"""
    digest = sha256_text(canonical_json(config))
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{digest[:4]}{secrets.token_hex(2)}"
    run_dir = ws.root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(config_toml, encoding="utf-8")

    rev, dirty = git_info()
    lock = REPO_ROOT / "uv.lock"
    started = now()
    meta = {
        "id": run_id,
        "kind": kind,
        "status": "running",
        "seed": seed,
        "config_digest": digest,
        "git_rev": rev,
        "git_dirty": dirty,
        "dataset_fp": dataset_fp,
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lock_sha256": sha256_file(lock) if lock.exists() else None,
        "argv": sys.argv,
        "started_at": started,
        "finished_at": None,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ws.db.execute(
        "INSERT INTO runs (id, kind, status, config_json, git_rev, git_dirty, dataset_fp, seed, started_at)"
        " VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            kind,
            canonical_json(config),
            rev,
            None if dirty is None else int(dirty),
            dataset_fp,
            seed,
            started,
        ),
    )
    ws.db.commit()

    run = Run(run_id, kind, run_dir, ws)
    handler = add_log_file(run_dir / "log.txt")
    try:
        yield run
    except KeyboardInterrupt:
        run._finish("interrupted")
        raise
    except BaseException:
        run._finish("failed")
        raise
    else:
        run._finish("done")
    finally:
        remove_log_file(handler)


@contextmanager
def resume_run(ws: Workspace, run_id: str, *, allow_done: bool = False) -> Iterator[Run]:
    """接着一次没跑完的运行继续：沿用它的目录和记录，状态改回 running。

    已完成的运行默认不许重新打开；对话可以接着聊，所以它传 allow_done=True。
    """
    row = ws.db.execute("SELECT kind, status FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"没有编号为 {run_id} 的运行")
    if row["status"] == "done" and not allow_done:
        raise ConflictError(f"运行 {run_id} 已经完成，不需要续跑")
    run = Run(run_id, row["kind"], ws.root / "runs" / run_id, ws)
    ws.db.execute("UPDATE runs SET status = 'running', finished_at = NULL WHERE id = ?", (run_id,))
    ws.db.commit()
    run.update_meta(status="running", finished_at=None, resumed_at=now())
    handler = add_log_file(run.dir / "log.txt")
    try:
        yield run
    except KeyboardInterrupt:
        run._finish("interrupted")
        raise
    except BaseException:
        run._finish("failed")
        raise
    else:
        run._finish("done")
    finally:
        remove_log_file(handler)


def truncate_metrics(run_dir: Path, max_step: int) -> None:
    """丢掉步数大于 max_step 的指标。续训从检查点的步数重新开始，之后的记录会重新产生。"""
    kept = [m for m in read_metrics(run_dir) if m["step"] <= max_step]
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in kept), encoding="utf-8")


def mark_stale_runs(ws: Workspace) -> int:
    """把遗留的 running 记录标为 interrupted（进程被强杀时来不及改状态）。返回改了几条。"""
    cur = ws.db.execute(
        "UPDATE runs SET status = 'interrupted', finished_at = ? WHERE status = 'running'", (now(),)
    )
    ws.db.commit()
    return cur.rowcount
