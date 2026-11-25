"""评测记录与逐回合结果。（策略，仿真器，评测集，步数上限，n_exec，种子）唯一确定一次评测。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from roboarena.store.workspace import Workspace

EvalKey = tuple[int, str, int, int, int, int]  # policy_id, sim, suite_id, max_steps, n_exec, seed
EpisodeRow = tuple[int, bool, float, int]  # init_index, success, max_progress, steps

_WHERE = "policy_id = ? AND sim = ? AND suite_id = ? AND max_steps = ? AND n_exec = ? AND seed = ?"


def find_eval(ws: Workspace, key: EvalKey) -> sqlite3.Row | None:
    return ws.db.execute(f"SELECT * FROM evals WHERE {_WHERE}", key).fetchone()


def create_eval(ws: Workspace, key: EvalKey, run_id: str | None) -> int:
    cur = ws.db.execute(
        "INSERT INTO evals (policy_id, sim, suite_id, max_steps, n_exec, seed, run_id, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'running')",
        (*key, run_id),
    )
    ws.db.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def delete_eval(ws: Workspace, eval_id: int) -> None:
    ws.db.execute("DELETE FROM evals WHERE id = ?", (eval_id,))
    ws.db.commit()


def save_episodes(ws: Workspace, eval_id: int, episodes: Iterable[EpisodeRow]) -> None:
    """写入逐回合结果并把评测标为完成。两步在同一个事务里。"""
    with ws.db:
        ws.db.executemany(
            "INSERT INTO episodes (eval_id, init_index, success, max_progress, steps) VALUES (?, ?, ?, ?, ?)",
            [(eval_id, i, int(s), float(p), int(n)) for i, s, p, n in episodes],
        )
        ws.db.execute("UPDATE evals SET status = 'done' WHERE id = ?", (eval_id,))


def episodes_of(ws: Workspace, eval_id: int) -> list[sqlite3.Row]:
    return ws.db.execute(
        "SELECT * FROM episodes WHERE eval_id = ? ORDER BY init_index", (eval_id,)
    ).fetchall()


def done_evals(ws: Workspace, suite_id: int, sim: str) -> list[sqlite3.Row]:
    """某个评测集、某个仿真器下已完成的评测，带策略名称。"""
    return ws.db.execute(
        "SELECT e.*, a.name AS policy FROM evals e JOIN artifacts a ON a.id = e.policy_id"
        " WHERE e.suite_id = ? AND e.sim = ? AND e.status = 'done' ORDER BY e.id",
        (suite_id, sim),
    ).fetchall()
