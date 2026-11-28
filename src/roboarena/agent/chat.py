"""对话的记忆（FR-G5）：一次对话是一条 chat 类型的运行记录，可以接着上次继续。

运行目录里：
    session.json     完整的对话状态，每轮结束后重写；接着聊时从它恢复
    turns.jsonl      每轮一行：用户的话、回答、工具调用与结果、用量、校验结果
    llm_calls.jsonl  每一次模型调用（网关记的）
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from roboarena.agent.loop import Confirm, Role, Session, Turn, run_turn
from roboarena.agent.tools import ToolContext
from roboarena.config import Config
from roboarena.core.errors import NotFoundError, RoboArenaError
from roboarena.llm.gateway import Gateway
from roboarena.store.db import now
from roboarena.store.runs import Run, resume_run, start_run
from roboarena.store.workspace import Workspace


@dataclass
class Chat:
    run: Run
    ctx: ToolContext
    gateway: Gateway
    model: str
    session: Session
    turns: int

    def say(self, text: str, *, confirm: Confirm | None = None, role: Role = "write") -> Turn:
        """处理用户的一句话并存盘。中途出错时这一轮不留痕迹：对话还停在上一轮结束的地方。"""
        trial = copy.deepcopy(self.session)
        turn = run_turn(
            self.ctx,
            self.gateway,
            trial,
            text,
            model=self.model,
            role=role,
            confirm=confirm,
            seed=self.ctx.cfg.seed,
        )
        self.session = trial
        self.turns += 1
        record = {
            "ts": now(),
            "model": self.model,
            "user": text,
            "answer": turn.answer,
            "stopped": turn.stopped,
            "steps": turn.steps,
            "retried": turn.retried,
            "numbers": len(turn.verdict.mentions),
            "unsupported": [m.text for m in turn.verdict.unsupported],
            "tokens_in": turn.tokens_in,
            "tokens_out": turn.tokens_out,
            "latency_ms": round(turn.latency_ms, 3),
            "events": [asdict(e) for e in turn.events],
        }
        with open(self.run.dir / "turns.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        tmp = self.run.dir / "session.json.tmp"
        tmp.write_text(json.dumps(self.session.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.run.dir / "session.json")  # 写到一半被打断也不会留下半个文件
        self.run.update_meta(turns=self.turns)
        return turn


@contextmanager
def open_chat(
    ws: Workspace, cfg: Config, gateway: Gateway, *, resume: str | None = None, model: str | None = None
) -> Iterator[Chat]:
    """开始一次对话，或接着编号为 resume 的那次继续。接着聊时默认沿用当时的模型。"""
    if resume is None:
        name = model or cfg.llm.default
        session, turns = Session.new(), 0
        manager = start_run(ws, "chat", config=cfg.to_dict(), config_toml=cfg.to_toml(), seed=cfg.seed)
    else:
        row = ws.db.execute("SELECT id, kind FROM runs WHERE id = ?", (resume,)).fetchone()
        if row is None:
            raise NotFoundError(f"没有编号为 {resume} 的运行")
        if row["kind"] != "chat":
            raise RoboArenaError(f"运行 {resume} 不是一次对话（类型是 {row['kind']}）")
        run_dir = ws.root / "runs" / row["id"]
        if not (run_dir / "session.json").exists():
            raise NotFoundError(f"运行 {resume} 没有可以接着聊的对话记录")
        session = Session.from_dict(json.loads((run_dir / "session.json").read_text(encoding="utf-8")))
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        name, turns = model or meta.get("model") or cfg.llm.default, int(meta.get("turns", 0))
        manager = resume_run(ws, row["id"], allow_done=True)
    with manager as run:
        run.update_meta(model=name, turns=turns)
        previous = gateway.calls_path
        gateway.calls_path = run.dir / "llm_calls.jsonl"
        try:
            yield Chat(run, ToolContext(ws, cfg), gateway, name, session, turns)
        finally:
            gateway.calls_path = previous
