import json
from pathlib import Path

import numpy as np
import pytest

from roboarena.cli import main
from roboarena.config import load_config
from roboarena.core.errors import ConfigError, ConflictError, IntegrityError, NotFoundError
from roboarena.data.collect import collect, noisy_replay_actions, random_walk_actions
from roboarena.data.dataset import SPLITS, build_dataset, load_dataset, split_groups, verify_dataset
from roboarena.data.episodes import episode_digest, load_episode, save_episode
from roboarena.data.validate import check_episode, check_replay
from roboarena.eval import service
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task

from .conftest import QUICK, REPO

TOY = get_task("toy")
PUSHT_RAW = REPO / "workspace" / "cache" / "pusht" / "raw"


def _toy_episodes(n=3, steps=6, seed=0):
    rng = np.random.default_rng(seed)
    inits = TOY.sample_inits(n, seed)
    seqs = [random_walk_actions(TOY.spec, steps, rng) for _ in range(n)]
    metas = [{"source": "random_walk", "index": i, "group": f"walk-{i}", "labeled": True} for i in range(n)]
    return collect(TOY, inits, seqs, metas, batch=2)


def _failed(ep, spec=TOY.spec):
    return {name for name, message in check_episode(spec, ep).items() if message is not None}


# ---- 回合文件 ----


def test_episode_round_trip_and_digest(tmp_path):
    ep = _toy_episodes(1)[0]
    save_episode(tmp_path / "e.npz", ep)
    back = load_episode(tmp_path / "e.npz")
    assert back.meta == ep.meta and set(back.labels) == set(ep.labels)
    assert np.array_equal(back.frames, ep.frames) and back.frames.dtype == np.uint8
    assert np.array_equal(back.actions, ep.actions) and back.success.dtype == bool
    assert episode_digest(back) == episode_digest(ep)
    assert list(tmp_path.iterdir()) == [tmp_path / "e.npz"]  # 没有留下临时文件

    back.meta["note"] = "描述变了，数据没变"
    assert episode_digest(back) == episode_digest(ep)
    back.frames = back.frames.copy()
    back.frames[2, 0, 0, 0] ^= 1  # 改动一个像素的一位
    assert episode_digest(back) != episode_digest(ep)


def test_npz_bytes_are_not_stable_but_digest_is(tmp_path):
    """直接对 .npz 求哈希不可靠的原因：同样的内容压缩两次，digest 相同就够了。"""
    ep = _toy_episodes(1)[0]
    save_episode(tmp_path / "a.npz", ep)
    save_episode(tmp_path / "b.npz", ep)
    assert episode_digest(load_episode(tmp_path / "a.npz")) == episode_digest(
        load_episode(tmp_path / "b.npz")
    )


# ---- 采集 ----


def test_action_generators_stay_in_range_and_are_seeded():
    spec = get_task("pusht").spec
    walk = random_walk_actions(spec, 200, np.random.default_rng(0))
    assert walk.shape == (200, 2) and walk.dtype == np.float32
    assert (walk >= spec.action_low).all() and (walk <= spec.action_high).all()
    assert np.array_equal(walk, random_walk_actions(spec, 200, np.random.default_rng(0)))
    assert np.abs(np.diff(walk, axis=0)).max() < 0.5 * 512  # 折线移动，不是每步瞬移

    base = np.full((50, 2), 500.0, dtype=np.float32)
    noisy = noisy_replay_actions(spec, base, 0.05, np.random.default_rng(0))
    assert noisy.max() <= 512 and not np.array_equal(noisy, base)
    assert np.array_equal(noisy_replay_actions(spec, base, 0.0, np.random.default_rng(0)), base)


def test_collect_handles_different_lengths_and_records_inits():
    rng = np.random.default_rng(1)
    inits = TOY.sample_inits(3, seed=1)
    seqs = [random_walk_actions(TOY.spec, n, rng) for n in (4, 9, 6)]
    metas = [{"source": "random_walk", "index": i, "group": f"walk-{i}", "labeled": True} for i in range(3)]
    together = collect(TOY, inits, seqs, metas, batch=3)
    assert [len(ep) for ep in together] == [4, 9, 6]
    for ep, init, seq in zip(together, inits, seqs, strict=True):
        assert ep.frames.shape[0] == len(seq) + 1 and np.array_equal(ep.actions, seq)
        assert TOY.decode_init(ep.meta["init"]) == init
        assert _failed(ep) == set() and check_replay(TOY, ep) is None
    alone = collect(TOY, inits, seqs, metas, batch=1)  # 批大小不影响结果
    assert [episode_digest(e) for e in alone] == [episode_digest(e) for e in together]


# ---- 校验：故意破坏数据，检查能报错 ----


def test_checks_pass_on_good_data():
    for ep in _toy_episodes():
        assert _failed(ep) == set() and check_replay(TOY, ep) is None


def test_schema_check_catches_shape_and_dtype_problems():
    ep = _toy_episodes(1)[0]
    ep.actions = ep.actions[:-1]  # 动作少一个：不再是 T 个动作配 T + 1 帧
    assert _failed(ep) == {"形状与类型"}
    ep = _toy_episodes(1)[0]
    ep.frames = ep.frames.astype(np.float32)
    assert _failed(ep) == {"形状与类型"}
    ep = _toy_episodes(1)[0]
    del ep.labels["agent_pos"]
    assert "标签字段" in (check_episode(TOY.spec, ep)["形状与类型"] or "")
    assert _failed(_toy_episodes(1)[0], spec=get_task("pusht").spec) == {"形状与类型"}  # 拿错了任务


def test_range_and_value_checks():
    ep = _toy_episodes(1)[0]
    ep.actions[2, 0] = 1.5
    assert _failed(ep) == {"动作范围"}
    ep.actions[2, 0] = np.nan
    assert _failed(ep) == {"动作范围"}

    ep = _toy_episodes(1)[0]
    ep.progress[1] = 1.2
    assert _failed(ep) == {"数值范围"}
    ep = _toy_episodes(1)[0]
    ep.frames[:] = 255
    assert _failed(ep) == {"数值范围"}

    ep = _toy_episodes(1)[0]
    ep.progress[0] = np.nan
    assert _failed(ep) == {"标签完整"}
    ep.meta["labeled"] = False  # 不声称带标签，NaN 就是允许的
    assert _failed(ep) == set()


def test_replay_check_catches_misaligned_actions():
    ep = _toy_episodes(1, steps=8)[0]
    shifted = np.roll(ep.actions, 1, axis=0)  # 动作整体错开一步
    ep.actions = shifted
    assert _failed(ep) == set()  # 形状和范围都没问题，只有重放能发现
    assert "画面与记录不同" in (check_replay(TOY, ep) or "")

    ep = _toy_episodes(1)[0]
    ep.frames = ep.frames[::-1].copy()
    assert "首帧" in (check_replay(TOY, ep) or "")
    ep = _toy_episodes(1)[0]
    ep.progress[3] += 0.01
    assert "进度" in (check_replay(TOY, ep) or "")
    ep.meta["init"] = None
    assert check_replay(TOY, ep) is None  # 没有初始状态的回合无法重放，不算失败


# ---- 切分 ----


def test_split_keeps_groups_together_and_mixes_sources():
    groups = [f"demo-{i}" for i in range(20)] * 3 + [f"walk-{i}" for i in range(10)]
    a = split_groups(groups, 0.2, 0.1, np.random.default_rng(0))
    assert a == split_groups(groups, 0.2, 0.1, np.random.default_rng(0))
    assert a != split_groups(groups, 0.2, 0.1, np.random.default_rng(1))
    assert set(a) == set(groups)
    count = lambda prefix, split: sum(g.startswith(prefix) and s == split for g, s in a.items())  # noqa: E731
    assert (count("demo-", "train"), count("demo-", "val"), count("demo-", "test")) == (14, 4, 2)
    assert (count("walk-", "train"), count("walk-", "val"), count("walk-", "test")) == (7, 2, 1)


# ---- 数据集 ----


@pytest.fixture
def built(cfg):
    with Workspace(cfg.workspace.root) as ws:
        yield ws, build_dataset(ws, cfg)


def test_build_writes_manifest_report_and_run(built, cfg):
    ws, ds = built
    m = ds.manifest
    assert (m["task"], m["name"], m["n_episodes"]) == ("toy", "main", 24)
    assert {e["source"] for e in m["episodes"]} == {"demo", "replay", "random_walk"}
    assert m["n_frames"] == sum(e["length"] + 1 for e in m["episodes"])
    assert m["replay_checked"] == {"replay": 8, "random_walk": 8}
    assert all((ds.dir / e["file"]).exists() for e in m["episodes"])
    assert (
        "按组" in (ds.dir / "REPORT.md").read_text()
        and m["fingerprint"] in (ds.dir / "REPORT.md").read_text()
    )
    run = ws.db.execute("SELECT * FROM runs").fetchone()
    assert (run["kind"], run["status"], run["dataset_fp"]) == ("data", "done", ds.fingerprint)
    assert load_dataset(ws, "toy", "main").manifest == m
    assert verify_dataset(ds, TOY) == []


def test_split_has_no_group_leakage_and_stats_use_train_only(built):
    _, ds = built
    groups = {s: {e["group"] for e in ds.entries(s)} for s in SPLITS}
    assert not (
        groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]
    )
    assert sum(len(ds.entries(s)) for s in SPLITS) == 24 and all(ds.entries(s) for s in SPLITS)
    # 一条演示和由它加噪重放的回合在同一个子集
    replay_groups = {e["group"] for e in ds.entries() if e["source"] == "replay"}
    assert replay_groups <= {e["group"] for e in ds.entries() if e["source"] == "demo"}

    train_actions = np.concatenate([ep.actions for ep in ds.episodes("train")])
    assert ds.manifest["stats"]["actions"]["mean"] == pytest.approx(train_actions.mean(axis=0), abs=1e-6)
    assert ds.manifest["stats"]["actions"]["std"] == pytest.approx(train_actions.std(axis=0), abs=1e-6)
    all_actions = np.concatenate([ep.actions for ep in ds.episodes()])
    assert ds.manifest["stats"]["actions"]["mean"] != pytest.approx(all_actions.mean(axis=0), abs=1e-6)


def test_build_is_reproducible_and_seed_sensitive(built, cfg, tmp_path):
    _, ds = built
    again = load_config(QUICK, [f'workspace.root="{tmp_path / "other"}"'])
    with Workspace(again.workspace.root) as ws:
        assert build_dataset(ws, again).fingerprint == ds.fingerprint
    again.seed = 1
    with Workspace(again.workspace.root) as ws:
        with pytest.raises(ConflictError, match="--force"):
            build_dataset(ws, again)
        other = build_dataset(ws, again, force=True)
        assert other.fingerprint != ds.fingerprint and other.manifest["splits"] != ds.manifest["splits"]


def test_verify_detects_tampering(built):
    _, ds = built
    entry = ds.entries()[10]
    path = ds.dir / entry["file"]
    ep = load_episode(path)
    ep.frames[1, 0, 0, 0] ^= 1
    save_episode(path, ep)
    problems = verify_dataset(ds, TOY)
    assert any(entry["file"] in p and "摘要不一致" in p for p in problems)
    assert any("指纹" in p for p in problems)

    path.unlink()
    assert any("文件不见了" in p for p in verify_dataset(ds, TOY))

    ds.manifest["splits"]["val"].append(ds.manifest["splits"]["train"][0])  # 同一个回合进了两个子集
    assert any("恰好覆盖" in p for p in verify_dataset(ds, TOY))


def test_build_fails_cleanly_when_a_check_fails(cfg, monkeypatch):
    import roboarena.data.dataset as module

    real = module.collect

    def corrupting_collect(task, inits, seqs, metas, batch=16):
        episodes = real(task, inits, seqs, metas, batch)
        episodes[0].actions = np.roll(episodes[0].actions, 1, axis=0)  # 让第一个回合的动作错位
        return episodes

    monkeypatch.setattr(module, "collect", corrupting_collect)
    with Workspace(cfg.workspace.root) as ws:
        with pytest.raises(IntegrityError, match="可重放"):
            build_dataset(ws, cfg)
        assert not (ws.root / "datasets" / "toy" / "main").exists()  # 不留下半成品
        assert ws.db.execute("SELECT status FROM runs").fetchone()[0] == "failed"
        with pytest.raises(NotFoundError, match="data build"):
            load_dataset(ws, "toy", "main")


def test_replay_needs_demos(cfg):
    cfg.data.n_demos = 0
    with Workspace(cfg.workspace.root) as ws, pytest.raises(ConfigError, match="n_replay"):
        build_dataset(ws, cfg)


def test_suite_inits_do_not_overlap_with_training_data(built, cfg):
    ws, ds = built
    suite = service.create_suite(ws, cfg)
    as_key = lambda d: json.dumps(d, sort_keys=True)  # noqa: E731
    assert not {as_key(i) for i in suite.inits} & {as_key(e["init"]) for e in ds.entries()}


def test_cli_data_commands(tmp_path, capsys):
    base = ["--config", str(QUICK), "--set", f'workspace.root="{tmp_path / "ws"}"']
    assert main(["data", "verify", *base]) == 1 and "data build" in capsys.readouterr().err
    assert main(["data", "build", *base]) == 0 and "24 个回合" in capsys.readouterr().out
    assert main(["data", "build", *base]) == 1 and "--force" in capsys.readouterr().err
    assert main(["data", "build", "--force", *base]) == 0
    capsys.readouterr()
    assert main(["data", "info", *base]) == 0 and "## 来源" in capsys.readouterr().out
    assert main(["data", "verify", *base]) == 0 and "核对通过" in capsys.readouterr().out
    victim = next((tmp_path / "ws" / "datasets" / "toy" / "main" / "episodes").iterdir())
    victim.unlink()
    assert main(["data", "verify", *base]) == 1 and "文件不见了" in capsys.readouterr().err


# ---- PushT 人类演示（需要已下载的原始文件；没有就跳过，测试不联网）----

needs_demos = pytest.mark.skipif(not PUSHT_RAW.exists(), reason="没有缓存的 PushT 演示数据")


@needs_demos
def test_pusht_demos_facts_and_alignment():
    task = get_task("pusht")
    episodes = task.demos(REPO / "workspace" / "cache")
    assert len(episodes) == 206 and sum(ep.frames.shape[0] for ep in episodes) == 25650
    assert all(_failed(ep, task.spec) == set() for ep in episodes[:20])
    ep = episodes[0]
    assert np.isnan(ep.labels["block_pose"]).all() and np.isnan(ep.progress[0]) and ep.meta["init"] is None

    # 动作对齐：从记录的推杆位置出发执行 a_t，应该到达记录的下一个推杆位置（块放到远处，不干扰推杆）
    from gym_pusht.envs.pusht import PushTEnv

    env = PushTEnv(obs_type="state")
    agent = ep.labels["agent_pos"]
    env.reset(options={"reset_to_state": [agent[0, 0], agent[0, 1], -2000, -2000, 0]})
    for t in range(20):
        env.step(ep.actions[t])
        assert np.allclose(env.agent.position, agent[t + 1], atol=1e-3)


@needs_demos
def test_pusht_demos_limit_is_a_prefix():
    task = get_task("pusht")
    few = task.demos(Path(REPO / "workspace" / "cache"), limit=3)
    assert len(few) == 3 and [e.meta["index"] for e in few] == [0, 1, 2]
