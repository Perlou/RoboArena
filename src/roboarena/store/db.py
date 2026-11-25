"""SQLite 表结构与连接。用标准库 sqlite3，不引入 ORM。

排行榜、秩相关等汇总结果不落表，每次由 episodes 现算：统计方法改了不需要迁移数据。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

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


def connect(path: Path) -> sqlite3.Connection:
    """打开数据库；新库会建表。表结构版本记在 PRAGMA user_version 里。"""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    elif version != SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"数据库 {path} 的表结构版本是 {version}，本程序需要 {SCHEMA_VERSION}")
    return conn
