import json
import sqlite3

import pytest

from roboarena.core.errors import ConflictError, IntegrityError, NotFoundError
from roboarena.store import evals as eval_store
from roboarena.store.registry import checkpoint_path, get_artifact, list_artifacts, register_artifact
from roboarena.store.runs import mark_stale_runs, read_metrics, start_run
from roboarena.store.suites import load_suite, save_suite
from roboarena.store.workspace import Workspace

CONFIG = {"seed": 0, "eval": {"suite": "main"}}


def _run(ws, kind="eval"):
    return start_run(ws, kind, config=CONFIG, config_toml="seed = 0\n", seed=0)


def test_workspace_creates_layout_and_reopens(tmp_path):
    with Workspace(tmp_path / "w") as ws:
        tables = {r[0] for r in ws.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"runs", "artifacts", "suites", "evals", "episodes", "jobs", "tokens"} <= tables
        assert (ws.root / "runs").is_dir()
    with Workspace(tmp_path / "w") as ws:  # 第二次打开不重新建表
        assert ws.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_run_leaves_a_complete_record(ws):
    with _run(ws) as run:
        run.log_metric(0, "loss", 1.5)
        run.log_metric(1, "loss", 0.5)
        assert ws.db.execute("SELECT status FROM runs WHERE id = ?", (run.id,)).fetchone()[0] == "running"
    row = ws.db.execute("SELECT * FROM runs WHERE id = ?", (run.id,)).fetchone()
    assert row["status"] == "done" and row["finished_at"] is not None
    assert json.loads(row["config_json"]) == CONFIG
    assert (run.dir / "config.toml").read_text() == "seed = 0\n"
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "done" and meta["seed"] == 0
    assert {"git_rev", "git_dirty", "lock_sha256", "python", "started_at", "finished_at"} <= set(meta)
    assert read_metrics(run.dir) == [
        {"step": 0, "name": "loss", "value": 1.5},
        {"step": 1, "name": "loss", "value": 0.5},
    ]
    assert (run.dir / "log.txt").exists()


def test_run_status_on_error_and_interrupt(ws):
    with pytest.raises(ValueError):
        with _run(ws) as failed:
            raise ValueError("boom")
    with pytest.raises(KeyboardInterrupt):
        with _run(ws) as interrupted:
            raise KeyboardInterrupt
    status = dict(ws.db.execute("SELECT id, status FROM runs").fetchall())
    assert status == {failed.id: "failed", interrupted.id: "interrupted"}


def test_run_ids_do_not_collide(ws):
    ids = set()
    for _ in range(5):
        with _run(ws) as run:
            ids.add(run.id)
    assert len(ids) == 5


def test_stale_runs_are_marked_interrupted(ws):
    ws.db.execute(
        "INSERT INTO runs (id, kind, status, config_json, started_at) VALUES ('r', 'eval', 'running', '{}', 't')"
    )
    assert mark_stale_runs(ws) == 1
    assert ws.db.execute("SELECT status FROM runs").fetchone()[0] == "interrupted"


def test_artifact_with_checkpoint_is_content_addressed_and_verified(ws, tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"weights-v1")
    art = register_artifact(
        ws, task="toy", name="bc-1", kind="policy", backend="bc", params={"chunk": 4}, checkpoint=ckpt
    )
    assert art.params == {"chunk": 4} and art.path is not None
    stored = checkpoint_path(ws, art)
    assert stored is not None and stored.read_bytes() == b"weights-v1"
    assert art.sha256 in art.path
    ckpt.unlink()  # 登记时复制了一份，原文件删掉不影响
    assert checkpoint_path(ws, get_artifact(ws, "bc-1")) == stored

    stored.write_bytes(b"weights-tampered")
    with pytest.raises(IntegrityError, match="不一致"):
        checkpoint_path(ws, art)
    stored.unlink()
    with pytest.raises(IntegrityError, match="不见了"):
        checkpoint_path(ws, art)


def test_artifact_without_checkpoint(ws):
    a = register_artifact(ws, task="toy", name="random", kind="policy", backend="random", params={})
    b = register_artifact(
        ws, task="toy", name="random-4", kind="policy", backend="random", params={"chunk": 4}
    )
    assert a.path is None and checkpoint_path(ws, a) is None
    assert a.sha256 != b.sha256


def test_artifact_names_are_unique_and_lookup_checks_kind(ws):
    register_artifact(ws, task="toy", name="x", kind="policy", backend="random", params={})
    with pytest.raises(ConflictError):
        register_artifact(ws, task="toy", name="x", kind="worldmodel", backend="latent", params={})
    with pytest.raises(NotFoundError):
        get_artifact(ws, "y")
    with pytest.raises(NotFoundError, match="世界模型"):
        get_artifact(ws, "x", kind="worldmodel")
    register_artifact(ws, task="pusht", name="wm", kind="worldmodel", backend="latent", params={})
    assert [a.name for a in list_artifacts(ws, task="toy")] == ["x"]
    assert [a.name for a in list_artifacts(ws, kind="worldmodel")] == ["wm"]
    assert len(list_artifacts(ws)) == 2


def test_suite_round_trip_and_integrity(ws):
    inits = [{"agent": [0.1, 0.2], "goal": [0.8, 0.9]}, {"agent": [0.5, 0.5], "goal": [0.2, 0.2]}]
    saved = save_suite(ws, "toy", "main", inits)
    loaded = load_suite(ws, "toy", "main")
    assert loaded == saved and loaded.inits == inits
    with pytest.raises(ConflictError):
        save_suite(ws, "toy", "main", inits)
    save_suite(ws, "pusht", "main", inits)  # 不同任务可以有同名评测集
    assert load_suite(ws, "pusht", "main").fingerprint != saved.fingerprint
    with pytest.raises(NotFoundError):
        load_suite(ws, "toy", "other")

    path = ws.root / "suites" / "toy" / "main.json"
    payload = json.loads(path.read_text())
    payload["inits"][0]["goal"] = [0.0, 0.0]
    path.write_text(json.dumps(payload))
    with pytest.raises(IntegrityError):
        load_suite(ws, "toy", "main")


def test_eval_key_is_unique_and_episodes_cascade(ws):
    art = register_artifact(ws, task="toy", name="p", kind="policy", backend="random", params={})
    suite = save_suite(ws, "toy", "main", [{"a": 1}, {"a": 2}])
    key = (art.id, "real", suite.id, 30, 1, 0)
    assert eval_store.find_eval(ws, key) is None
    eval_id = eval_store.create_eval(ws, key, None)
    with pytest.raises(sqlite3.IntegrityError):
        eval_store.create_eval(ws, key, None)
    ws.db.rollback()
    assert (
        eval_store.create_eval(ws, (art.id, "real", suite.id, 30, 1, 1), None) != eval_id
    )  # 换个种子是另一次评测

    assert eval_store.done_evals(ws, suite.id, "real") == []  # 还没写结果，不算完成
    eval_store.save_episodes(ws, eval_id, [(0, True, 0.97, 12), (1, False, 0.4, 30)])
    rows = eval_store.episodes_of(ws, eval_id)
    assert [(r["init_index"], r["success"], r["steps"]) for r in rows] == [(0, 1, 12), (1, 0, 30)]
    assert [e["policy"] for e in eval_store.done_evals(ws, suite.id, "real")] == ["p"]

    eval_store.delete_eval(ws, eval_id)
    assert eval_store.episodes_of(ws, eval_id) == []


# ---- 表结构迁移 ----

from roboarena.store import db as dbmod  # noqa: E402


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _old_db(path, version=1):
    """一个版本 1 的库，里面有一条运行记录。"""
    conn = sqlite3.connect(path)
    conn.executescript(dbmod.SCHEMA)
    conn.execute(
        "INSERT INTO runs (id, kind, status, config_json, started_at) VALUES ('r1', 'eval', 'done', '{}', 't')"
    )
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    conn.close()


def test_new_database_is_at_the_current_version_without_a_backup(tmp_path):
    conn = dbmod.connect(tmp_path / "a.db")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION == 2
    assert {"ai_evals", "ai_cases", "runs", "episodes"} <= _tables(conn)
    conn.close()
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".bak") == []


def test_version_1_database_is_backed_up_then_migrated(tmp_path):
    path = tmp_path / "a.db"
    _old_db(path)

    conn = dbmod.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert {"ai_evals", "ai_cases"} <= _tables(conn)
    assert conn.execute("SELECT id FROM runs").fetchall()[0]["id"] == "r1"  # 原有的数据还在
    conn.execute(
        "INSERT INTO runs (id, kind, status, config_json, started_at) VALUES ('r2', 'chat', 'done', '{}', 't')"
    )
    conn.commit()  # 迁移之后连接照常使用
    conn.close()

    backup = sqlite3.connect(tmp_path / "a.db.v1.bak")
    assert backup.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "ai_evals" not in _tables(backup)
    assert [r[0] for r in backup.execute("SELECT id FROM runs")] == ["r1"]
    backup.close()

    dbmod.connect(path).close()  # 再打开一次：不再迁移，也不动备份
    assert [p.name for p in tmp_path.glob("*.bak*")] == ["a.db.v1.bak"]


def test_failed_migration_leaves_the_old_version(tmp_path, monkeypatch):
    path = tmp_path / "a.db"
    _old_db(path)
    monkeypatch.setitem(
        dbmod.MIGRATIONS, 2, "CREATE TABLE ai_evals (id INTEGER); CREATE TABLE runs (id INTEGER);"
    )
    with pytest.raises(sqlite3.OperationalError):
        dbmod.connect(path)

    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "ai_evals" not in _tables(conn)  # 前一半建的表也撤销了
    conn.close()
    assert (tmp_path / "a.db.v1.bak").exists()

    monkeypatch.undo()
    conn = dbmod.connect(path)  # 修好之后重来：能迁移，早先的备份留着
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_database_from_a_newer_program_is_refused(tmp_path):
    path = tmp_path / "a.db"
    _old_db(path, version=99)
    with pytest.raises(RuntimeError, match="表结构版本是 99，比本程序认识的 2 新"):
        dbmod.connect(path)


def test_ai_eval_key_is_unique_and_cases_cascade(ws):
    insert = "INSERT INTO ai_evals (suite, suite_fp, model, setup, seed, status) VALUES ('toolcall', 'fp', 'm', 's', ?, 'done')"
    eval_id = ws.db.execute(insert, (0,)).lastrowid
    ws.db.execute(insert, (1,))
    with pytest.raises(sqlite3.IntegrityError):
        ws.db.execute(insert, (0,))
    ws.db.execute("INSERT INTO ai_cases VALUES (?, 'c1', 'f', 'seen', 1, '{}', 1.5, 10, 2, 1)", (eval_id,))
    ws.db.execute("DELETE FROM ai_evals WHERE id = ?", (eval_id,))
    assert ws.db.execute("SELECT COUNT(*) FROM ai_cases").fetchone()[0] == 0
