"""AI 评测的记录与逐用例结果。（用例集指纹，模型，配置，种子）唯一确定一次评测；汇总由明细现算。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any

from roboarena.store.workspace import Workspace

AiEvalKey = tuple[str, str, str, int]  # suite_fp, model, setup, seed
# case_id, family, bucket, passed, scores_json, latency_ms, tokens_in, tokens_out, steps
AiCaseRow = tuple[str, str, str, bool, str, float, int, int, int]

_WHERE = "suite_fp = ? AND model = ? AND setup = ? AND seed = ?"


def find_ai_eval(ws: Workspace, key: AiEvalKey) -> sqlite3.Row | None:
    return ws.db.execute(f"SELECT * FROM ai_evals WHERE {_WHERE}", key).fetchone()


def create_ai_eval(ws: Workspace, suite: str, key: AiEvalKey, run_id: str | None) -> int:
    cur = ws.db.execute(
        "INSERT INTO ai_evals (suite, suite_fp, model, setup, seed, run_id, status)"
        " VALUES (?, ?, ?, ?, ?, ?, 'running')",
        (suite, *key, run_id),
    )
    ws.db.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def delete_ai_eval(ws: Workspace, eval_id: int) -> None:
    ws.db.execute("DELETE FROM ai_evals WHERE id = ?", (eval_id,))
    ws.db.commit()


def save_ai_cases(ws: Workspace, eval_id: int, rows: Iterable[AiCaseRow]) -> None:
    """写入逐用例结果并把评测标为完成。两步在同一个事务里。"""
    with ws.db:
        ws.db.executemany(
            "INSERT INTO ai_cases (eval_id, case_id, family, bucket, passed, scores_json, latency_ms,"
            " tokens_in, tokens_out, steps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (eval_id, c, f, b, int(p), s, float(ms), int(ti), int(to), int(n))
                for c, f, b, p, s, ms, ti, to, n in rows
            ],
        )
        ws.db.execute("UPDATE ai_evals SET status = 'done' WHERE id = ?", (eval_id,))


def ai_case_rows(ws: Workspace, suite: str) -> list[dict[str, Any]]:
    """某个套件下已完成的评测的全部逐用例结果，带评测的键。"""
    rows = ws.db.execute(
        "SELECT e.suite_fp, e.model, e.setup, e.seed, c.* FROM ai_cases c JOIN ai_evals e ON e.id = c.eval_id"
        " WHERE e.suite = ? AND e.status = 'done' ORDER BY e.id, c.case_id",
        (suite,),
    ).fetchall()
    return [dict(r) for r in rows]
