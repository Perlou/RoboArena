"""SQLite 表结构与连接。用标准库 sqlite3，不引入 ORM。

排行榜、秩相关等汇总结果不落表，每次由 episodes 现算：统计方法改了不需要迁移数据。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 2

# 版本 1 的全部表。之后的改动只写在 MIGRATIONS 里：只新增，不改已有的表
SCHEMA = """
CREATE TABLE runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed', 'interrupted')),
    config_json TEXT NOT NULL,
    git_rev     TEXT,
    git_dirty   INTEGER,
    dataset_fp  TEXT,
    seed        INTEGER,
    started_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE artifacts (
    id          INTEGER PRIMARY KEY,
    task        TEXT NOT NULL,
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL CHECK (kind IN ('policy', 'worldmodel')),
    backend     TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    path        TEXT,
    run_id      TEXT REFERENCES runs (id),
    params_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE suites (
    id          INTEGER PRIMARY KEY,
    task        TEXT NOT NULL,
    name        TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    n_inits     INTEGER NOT NULL,
    path        TEXT NOT NULL,
    UNIQUE (task, name)
);

CREATE TABLE evals (
    id        INTEGER PRIMARY KEY,
    policy_id INTEGER NOT NULL REFERENCES artifacts (id),
    sim       TEXT NOT NULL,
    suite_id  INTEGER NOT NULL REFERENCES suites (id),
    max_steps INTEGER NOT NULL,
    n_exec    INTEGER NOT NULL,
    seed      INTEGER NOT NULL,
    run_id    TEXT REFERENCES runs (id),
    status    TEXT NOT NULL CHECK (status IN ('running', 'done')),
    UNIQUE (policy_id, sim, suite_id, max_steps, n_exec, seed)
);

CREATE TABLE episodes (
    eval_id      INTEGER NOT NULL REFERENCES evals (id) ON DELETE CASCADE,
    init_index   INTEGER NOT NULL,
    success      INTEGER NOT NULL,
    max_progress REAL NOT NULL,
    steps        INTEGER NOT NULL,
    PRIMARY KEY (eval_id, init_index)
);

CREATE TABLE jobs (
    id           INTEGER PRIMARY KEY,
    kind         TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status       TEXT NOT NULL,
    progress     REAL NOT NULL DEFAULT 0,
    error        TEXT,
    trace_id     TEXT,
    token_id     INTEGER REFERENCES tokens (id),
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE tokens (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    hash       TEXT NOT NULL UNIQUE,
    role       TEXT NOT NULL CHECK (role IN ('read', 'write')),
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
"""


def now() -> str:
    """当前时间，UTC，ISO 8601，精确到秒。数据库里的时间一律用它。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


# 版本 N-1 → N 的迁移。新库也是先建版本 1 的表，再依次执行这些
MIGRATIONS: dict[int, str] = {
    2: """
CREATE TABLE ai_evals (
    id       INTEGER PRIMARY KEY,
    suite    TEXT NOT NULL,
    suite_fp TEXT NOT NULL,
    model    TEXT NOT NULL,
    setup    TEXT NOT NULL,
    seed     INTEGER NOT NULL,
    run_id   TEXT REFERENCES runs (id),
    status   TEXT NOT NULL CHECK (status IN ('running', 'done')),
    UNIQUE (suite_fp, model, setup, seed)
);

CREATE TABLE ai_cases (
    eval_id     INTEGER NOT NULL REFERENCES ai_evals (id) ON DELETE CASCADE,
    case_id     TEXT NOT NULL,
    family      TEXT NOT NULL,
    bucket      TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    scores_json TEXT NOT NULL,
    latency_ms  REAL NOT NULL,
    tokens_in   INTEGER NOT NULL,
    tokens_out  INTEGER NOT NULL,
    steps       INTEGER NOT NULL,
    PRIMARY KEY (eval_id, case_id)
);
""",
}


def backup_path(path: Path, version: int) -> Path:
    return path.with_name(f"{path.name}.v{version}.bak")


def _backup(conn: sqlite3.Connection, dest: Path) -> None:
    """用 sqlite3 的 backup 接口复制整个库。库在 WAL 模式，直接复制文件不可靠。"""
    if dest.exists():  # 上次迁移到一半留下的备份更早，留着它
        return
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.unlink(missing_ok=True)
    target = sqlite3.connect(tmp)
    try:
        conn.backup(target)
    finally:
        target.close()
    tmp.replace(dest)


def _migrate(conn: sqlite3.Connection, version: int) -> None:
    """执行一步迁移。建表与改版本号在同一个事务里：中途失败时库还是原来的版本。"""
    conn.execute("BEGIN")
    try:
        for statement in MIGRATIONS[version].split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {version}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def connect(path: Path) -> sqlite3.Connection:
    """打开数据库；新库会建表，旧版本的库先备份成 <文件名>.v<旧版本>.bak 再迁移。

    表结构版本记在 PRAGMA user_version 里。
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(
            f"数据库 {path} 的表结构版本是 {version}，比本程序认识的 {SCHEMA_VERSION} 新；请更新程序"
        )
    if version == 0:
        conn.executescript(SCHEMA)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        version = 1
    elif version < SCHEMA_VERSION:
        _backup(conn, backup_path(path, version))
    previous = conn.isolation_level
    conn.isolation_level = None  # 迁移时自己管事务：sqlite3 默认不把建表语句放进事务
    try:
        for target in range(version + 1, SCHEMA_VERSION + 1):
            _migrate(conn, target)
    finally:
        conn.isolation_level = previous
    return conn
