"""模型与策略注册表。

带权重的条目：检查点按内容哈希存放在 artifacts/<前两位>/<哈希>/ 下，加载前重新核对哈希。
不带权重的条目（如随机策略）：只有构造参数，哈希取自参数本身。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboarena.core.errors import ConflictError, IntegrityError, NotFoundError
from roboarena.core.hashing import canonical_json, sha256_file, sha256_text
from roboarena.store.db import now
from roboarena.store.workspace import Workspace


@dataclass(frozen=True)
class Artifact:
    id: int
    task: str
    name: str
    kind: str  # "policy" 或 "worldmodel"
    backend: str
    sha256: str
    path: str | None  # 相对工作区；不带权重时为 None
    run_id: str | None
    params: dict[str, Any]
    created_at: str


def _from_row(row: sqlite3.Row) -> Artifact:
    return Artifact(
        id=row["id"],
        task=row["task"],
        name=row["name"],
        kind=row["kind"],
        backend=row["backend"],
        sha256=row["sha256"],
        path=row["path"],
        run_id=row["run_id"],
        params=json.loads(row["params_json"]),
        created_at=row["created_at"],
    )


def register_artifact(
    ws: Workspace,
    *,
    task: str,
    name: str,
    kind: str,
    backend: str,
    params: Mapping[str, Any],
    checkpoint: Path | None = None,
    run_id: str | None = None,
) -> Artifact:
    """登记一个策略或世界模型。名称全局唯一，重名报错。checkpoint 会被复制进工作区。"""
    if ws.db.execute("SELECT 1 FROM artifacts WHERE name = ?", (name,)).fetchone():
        raise ConflictError(f"名称已被占用：{name}")
    params_json = canonical_json(dict(params))
    rel: str | None = None
    if checkpoint is None:
        sha = sha256_text(f"{backend}|{params_json}")
    else:
        sha = sha256_file(checkpoint)
        dest = ws.root / "artifacts" / sha[:2] / sha / checkpoint.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, dest)
        rel = ws.rel(dest)
    ws.db.execute(
        "INSERT INTO artifacts (task, name, kind, backend, sha256, path, run_id, params_json, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task, name, kind, backend, sha, rel, run_id, params_json, now()),
    )
    ws.db.commit()
    return get_artifact(ws, name)


def get_artifact(ws: Workspace, name: str, kind: str | None = None) -> Artifact:
    row = ws.db.execute("SELECT * FROM artifacts WHERE name = ?", (name,)).fetchone()
    if row is None or (kind is not None and row["kind"] != kind):
        what = {"policy": "策略", "worldmodel": "世界模型"}.get(kind or "", "条目")
        raise NotFoundError(f"注册表里没有名为 {name} 的{what}")
    return _from_row(row)


def list_artifacts(ws: Workspace, task: str | None = None, kind: str | None = None) -> list[Artifact]:
    rows = ws.db.execute(
        "SELECT * FROM artifacts WHERE (?1 IS NULL OR task = ?1) AND (?2 IS NULL OR kind = ?2) ORDER BY id",
        (task, kind),
    ).fetchall()
    return [_from_row(r) for r in rows]


def checkpoint_path(ws: Workspace, artifact: Artifact) -> Path | None:
    """返回检查点的绝对路径，并核对内容哈希。不带权重的条目返回 None。"""
    if artifact.path is None:
        return None
    path = ws.abs(artifact.path)
    if not path.exists():
        raise IntegrityError(f"{artifact.name} 的检查点不见了：{artifact.path}")
    if sha256_file(path) != artifact.sha256:
        raise IntegrityError(f"{artifact.name} 的检查点内容与登记时不一致：{artifact.path}")
    return path
