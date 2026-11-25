"""评测集：一组固定的初始状态。初始状态由任务插件序列化成字典，这里只负责存取和指纹。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from roboarena.core.errors import ConflictError, IntegrityError, NotFoundError
from roboarena.core.hashing import canonical_json, sha256_text
from roboarena.store.workspace import Workspace


@dataclass(frozen=True)
class Suite:
    id: int
    task: str
    name: str
    fingerprint: str
    inits: list[dict[str, Any]]


def suite_fingerprint(task: str, inits: Sequence[dict[str, Any]]) -> str:
    return sha256_text(canonical_json({"task": task, "inits": list(inits)}))


def save_suite(ws: Workspace, task: str, name: str, inits: Sequence[dict[str, Any]]) -> Suite:
    if ws.db.execute("SELECT 1 FROM suites WHERE task = ? AND name = ?", (task, name)).fetchone():
        raise ConflictError(f"任务 {task} 已有名为 {name} 的评测集")
    fp = suite_fingerprint(task, inits)
    path = ws.root / "suites" / task / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "name": name, "fingerprint": fp, "inits": list(inits)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ws.db.execute(
        "INSERT INTO suites (task, name, fingerprint, n_inits, path) VALUES (?, ?, ?, ?, ?)",
        (task, name, fp, len(inits), ws.rel(path)),
    )
    ws.db.commit()
    return load_suite(ws, task, name)


def load_suite(ws: Workspace, task: str, name: str) -> Suite:
    row = ws.db.execute("SELECT * FROM suites WHERE task = ? AND name = ?", (task, name)).fetchone()
    if row is None:
        raise NotFoundError(f"任务 {task} 没有名为 {name} 的评测集，先运行 roboarena data suite")
    payload = json.loads(ws.abs(row["path"]).read_text(encoding="utf-8"))
    if suite_fingerprint(task, payload["inits"]) != row["fingerprint"]:
        raise IntegrityError(f"评测集 {task}/{name} 的内容与登记时不一致")
    return Suite(row["id"], task, name, row["fingerprint"], payload["inits"])


def has_suite(ws: Workspace, task: str, name: str) -> bool:
    return (
        ws.db.execute("SELECT 1 FROM suites WHERE task = ? AND name = ?", (task, name)).fetchone() is not None
    )
