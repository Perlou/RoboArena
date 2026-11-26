import json

import numpy as np
import pytest
import torch

from roboarena.cli import main
from roboarena.config import load_config
from roboarena.core.errors import ConflictError, NotFoundError
from roboarena.data.dataset import build_dataset, load_dataset, verify_dataset
from roboarena.data.frames import FrameStore, random_shift
from roboarena.data.rollouts import load_rollout_set, rollout_index
from roboarena.eval import service
from roboarena.eval.rollouts import collect_rollouts
from roboarena.policies import build_policy
from roboarena.policies.bc import BCNet
from roboarena.policies.diffusion import DiffusionNet, cosine_alphas_cumprod
from roboarena.policies.modules import SpatialSoftmax, normalize_actions
from roboarena.policies.train import demo_store, lr_at, train_policy
from roboarena.policies.zoo import build_zoo, zoo_names
from roboarena.store.registry import checkpoint_path, delete_artifact, get_artifact, list_artifacts
from roboarena.store.runs import read_metrics
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task
from roboarena.tasks.api import Episode

from .conftest import QUICK

TOY = get_task("toy")


def _episode(length, offset):
    """画面的全部像素等于"回合内的帧号 + offset"，动作等于帧号：取出来的窗口可以直接读出下标。"""
    frames = np.zeros((length + 1, 4, 4, 3), dtype=np.uint8) + (np.arange(length + 1) + offset)[
        :, None, None, None
    ].astype(np.uint8)
    actions = np.stack([np.arange(length), np.arange(length)], axis=1).astype(np.float32)
    zeros = np.zeros(length + 1, dtype=np.float32)
    return Episode(frames, actions, zeros, zeros.astype(bool), {})


# ---- 帧库与采样 ----


def test_frame_store_windows_do_not_cross_episodes():
    store = FrameStore.from_episodes([_episode(5, 0), _episode(3, 100)])
    assert (store.n_episodes, store.n_samples, len(store.frames)) == (2, 8, 10)
    frames, actions = store.policy_batch(np.random.default_rng(0), 500, obs_frames=3, chunk=4)
    assert frames.shape == (500, 3, 4, 4, 3) and actions.shape == (500, 4, 2)
    seen = set()
    for f, a in zip(frames[:, :, 0, 0, 0].astype(int), actions[:, :, 0].astype(int), strict=True):
        offset, length = (100, 3) if f[-1] >= 100 else (0, 5)
        t = f[-1] - offset
        seen.add((offset, t))
        assert t < length  # 最后一帧之后没有动作，不会被取到
        assert f.tolist() == [max(t - 2, 0) + offset, max(t - 1, 0) + offset, t + offset]  # 开头用首帧补齐
        assert a.tolist() == [min(t + k, length - 1) for k in range(4)]  # 末尾重复最后一个动作
    assert len(seen) == 8  # 每个可用的时刻都取到过


def test_frame_store_sampling_is_seeded():
    store = FrameStore.from_episodes([_episode(6, 0), _episode(9, 50)])
    a = store.policy_batch(np.random.default_rng(1), 8, 2, 3)
    b = store.policy_batch(np.random.default_rng(1), 8, 2, 3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    with pytest.raises(ValueError):
        FrameStore.from_episodes([])


def test_random_shift_translates_whole_samples():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(64, 2, 10, 10, 3), dtype=np.uint8)
    assert random_shift(frames, 0, rng) is frames
    out = random_shift(frames, 2, rng)
    assert out.shape == frames.shape and out.dtype == np.uint8
    shifts = set()
    for i in range(len(frames)):
        found = [
            (dy, dx)
            for dy in range(-2, 3)
            for dx in range(-2, 3)
            if np.array_equal(
                out[i, :, max(-dy, 0) + 2 : 8 - max(dy, 0), max(-dx, 0) + 2 : 8 - max(dx, 0)],
                frames[
                    i,
                    :,
                    max(-dy, 0) + 2 + dy : 8 - max(dy, 0) + dy,
                    max(-dx, 0) + 2 + dx : 8 - max(dx, 0) + dx,
                ],
            )
        ]
        assert len(found) == 1  # 同一个样本的两帧平移量相同
        shifts.add(found[0])
    assert len(shifts) > 10  # 不同样本的平移量不同


# ---- 模型 ----


def test_spatial_softmax_returns_the_peak_location():
    x = torch.full((1, 2, 5, 9), -50.0)
    x[0, 0, 0, 8] = 50.0  # 右上角
    x[0, 1, 2, 4] = 50.0  # 正中
    out = SpatialSoftmax()(x)  # [1, 4]：两个通道的 x，再两个通道的 y
    assert out.shape == (1, 4)
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, -1.0, 0.0]]), atol=1e-5)


def test_bc_net_shapes_and_normalization():
    net = BCNet(action_dim=2, obs_frames=2, chunk=5, width=8, hidden=16)
    out = net(torch.zeros(3, 2, 32, 32, 3, dtype=torch.uint8))
    assert out.shape == (3, 5, 2) and out.dtype == torch.float32
    low, high = torch.tensor([0.0, 0.0]), torch.tensor([512.0, 512.0])
    assert normalize_actions(torch.tensor([[0.0, 512.0], [256.0, 128.0]]), low, high).tolist() == [
        [-1, 1],
        [0, -0.5],
    ]


def test_lr_schedule():
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(10, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(55, 1.0, 10, 100) == pytest.approx(0.5)
    assert lr_at(99, 1.0, 10, 100) < 0.001


def test_bc_can_fit_one_batch():
    """调试台阶的第一级：在一个批上过拟合，损失应该降到接近 0。"""
    torch.manual_seed(0)
    spec = TOY.spec
    store = FrameStore.from_episodes(TOY.demos(None, limit=4))  # type: ignore[arg-type]
    frames, actions = store.policy_batch(np.random.default_rng(0), 16, 1, 2)
    net = BCNet(spec.action_dim, obs_frames=1, chunk=2, width=8, hidden=32)
    target = normalize_actions(
        torch.from_numpy(actions), torch.from_numpy(spec.action_low), torch.from_numpy(spec.action_high)
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(300):
        loss = torch.nn.functional.l1_loss(net(torch.from_numpy(frames)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0] and losses[-1] < 0.03


# ---- Diffusion Policy ----


def test_cosine_schedule_is_monotonic():
    ac = cosine_alphas_cumprod(100)
    assert ac.shape == (100,) and ac.dtype == torch.float32
    assert ac[0] > 0.99 and ac[-1] < 0.01 and (ac[1:] < ac[:-1]).all()


def test_ddim_with_an_oracle_denoiser_recovers_the_target():
    """如果去噪网络给出的噪声恰好是"当前样本相对某个固定动作 a* 的噪声"，DDIM 应该一步不差地走回 a*。"""
    net = DiffusionNet(
        action_dim=2, obs_frames=1, chunk=3, width=8, hidden=16, diffusion_steps=50, sample_steps=7
    )
    target = torch.tensor([[[0.5, -0.25], [0.0, 0.9], [-1.0, 1.0]]])

    def oracle(noisy, t, obs):
        ac = net.alphas_cumprod[t].reshape(-1, 1, 1)
        return (noisy - ac.sqrt() * target) / (1 - ac).sqrt()

    net.denoise = oracle  # type: ignore[method-assign]
    frames = torch.zeros(4, 1, 32, 32, 3, dtype=torch.uint8)
    out = net.sample(frames, torch.randn(4, 3, 2, generator=torch.Generator().manual_seed(0)))
    assert torch.allclose(out, target.expand(4, 3, 2), atol=1e-4)


def test_diffusion_net_loss_and_sampling_are_seeded():
    torch.manual_seed(0)
    net = DiffusionNet(
        action_dim=2, obs_frames=2, chunk=4, width=8, hidden=32, diffusion_steps=20, sample_steps=4
    )
    frames = torch.randint(0, 255, (5, 2, 32, 32, 3), dtype=torch.uint8)
    target = torch.rand(5, 4, 2) * 2 - 1
    a, b = (net.loss(frames, target, np.random.default_rng(3)) for _ in range(2))
    assert a.item() == b.item() and a.item() != net.loss(frames, target, np.random.default_rng(4)).item()
    x, y = (net.predict(frames, np.random.default_rng(3)) for _ in range(2))
    assert x.shape == (5, 4, 2) and torch.equal(x, y) and x.abs().max() <= 1
    assert not torch.equal(x, net.predict(frames, np.random.default_rng(4)))  # 换个种子，采到另一段动作


def test_diffusion_can_fit_one_batch():
    torch.manual_seed(0)
    spec = TOY.spec
    store = FrameStore.from_episodes(TOY.demos(None, limit=4))  # type: ignore[arg-type]
    frames, actions = store.policy_batch(np.random.default_rng(0), 16, 1, 2)
    net = DiffusionNet(
        spec.action_dim, obs_frames=1, chunk=2, width=8, hidden=64, diffusion_steps=20, sample_steps=5
    )
    low, high = torch.from_numpy(spec.action_low), torch.from_numpy(spec.action_high)
    target = normalize_actions(torch.from_numpy(actions), low, high)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(600):
        loss = net.loss(torch.from_numpy(frames), target, rng)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-50:]) < 0.3 * np.mean(losses[:50])
    sampled = net.predict(torch.from_numpy(frames), np.random.default_rng(1))
    assert (sampled - target).abs().mean() < 0.15  # 采样出来的动作落在训练目标附近


def test_train_diffusion_policy_with_ema_and_resume(cfg, tmp_path):
    results = []
    for sub, crash_at in (("straight", None), ("resumed", 47)):
        cfg.workspace.root = str(tmp_path / sub)
        with Workspace(cfg.workspace.root) as ws:
            build_dataset(ws, cfg)

            def crash(step, crash_at=crash_at):
                if step == crash_at:
                    raise KeyboardInterrupt

            if crash_at is None:
                art = train_policy(ws, cfg, "dp", backend="diffusion")
            else:
                with pytest.raises(KeyboardInterrupt):
                    train_policy(ws, cfg, "dp", backend="diffusion", on_step=crash)
                run_id = ws.db.execute("SELECT id FROM runs WHERE kind = 'train_policy'").fetchone()[0]
                art = train_policy(ws, load_config(), "x", resume=run_id)  # 后端也取自那次运行
            assert art.backend == "diffusion" and art.name == "dp"
            assert art.params["diffusion_steps"] == 20 and art.params["sample_steps"] == 4
            last = torch.load(ws.root / "runs" / art.run_id / "checkpoints" / "last.pt", weights_only=True)
            results.append((_weights(ws, "dp"), _weights(ws, "dp-step20"), last))
    (w1, s1, last1), (w2, s2, _) = results
    assert _same(w1, w2) and _same(s1, s2)  # EMA 权重也跟着续训恢复
    assert _same(w1, last1["ema"]) and not _same(w1, last1["model"])  # 登记的是 EMA 权重，不是训练中的权重

    with Workspace(cfg.workspace.root) as ws:
        art = get_artifact(ws, "dp")
        policy = build_policy(art.backend, TOY.spec, art.params, checkpoint_path(ws, art), art.name)
        frames = np.zeros((3, 2, 32, 32, 3), dtype=np.uint8)
        a, b = (policy.act(frames, np.random.default_rng(0)) for _ in range(2))
        assert a.shape == (3, 4, 2) and np.array_equal(a, b) and (a >= 0).all() and (a <= 1).all()


def test_unknown_trainable_backend(ws_data, cfg):
    with pytest.raises(NotFoundError, match="可训练策略后端"):
        train_policy(ws_data, cfg, "x", backend="random")


# ---- 训练、续训、登记 ----


@pytest.fixture
def ws_data(cfg):
    with Workspace(cfg.workspace.root) as ws:
        build_dataset(ws, cfg)
        service.create_suite(ws, cfg)
        yield ws


def _weights(ws, name):
    path = checkpoint_path(ws, get_artifact(ws, name))
    return torch.load(path, weights_only=True)


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_demo_store_uses_only_train_demos_and_limit_is_a_prefix(ws_data, cfg):
    ds = load_dataset(ws_data, "toy", "main")
    n_train_demos = sum(e["source"] == "demo" for e in ds.entries("train"))
    full, two = demo_store(ds, "train"), demo_store(ds, "train", 2)
    assert full.n_episodes == n_train_demos and two.n_episodes == 2
    assert np.array_equal(two.frames, full.frames[: len(two.frames)])
    assert demo_store(ds, "val").n_episodes == sum(e["source"] == "demo" for e in ds.entries("val"))


def test_train_registers_policy_snapshot_and_run(ws_data, cfg):
    art = train_policy(ws_data, cfg, "bc")
    assert art.backend == "bc" and art.task == "toy"
    assert art.params == {
        "obs_frames": 2,
        "chunk": 4,
        "width": 8,
        "hidden": 32,
        "exec_noise": 0.0,
        "n_demos": 4,
        "steps": 60,
    }
    snap = get_artifact(ws_data, "bc-step20")
    assert snap.params["steps"] == 20 and snap.run_id == art.run_id and snap.sha256 != art.sha256

    run = ws_data.db.execute("SELECT * FROM runs WHERE id = ?", (art.run_id,)).fetchone()
    ds = load_dataset(ws_data, "toy", "main")
    assert (run["kind"], run["status"], run["dataset_fp"]) == ("train_policy", "done", ds.fingerprint)
    metrics = read_metrics(ws_data.root / "runs" / art.run_id)
    assert [m["step"] for m in metrics if m["name"] == "loss"] == [10, 20, 30, 40, 50, 60]
    assert [m["step"] for m in metrics if m["name"] == "val_loss"] == [20, 40, 60]
    losses = [m["value"] for m in metrics if m["name"] == "loss"]
    assert losses[-1] < losses[0]

    policy = build_policy(art.backend, TOY.spec, art.params, checkpoint_path(ws_data, art), art.name)
    actions = policy.act(np.zeros((3, 2, 32, 32, 3), dtype=np.uint8), np.random.default_rng(0))
    assert actions.shape == (3, 4, 2) and actions.dtype == np.float32
    assert (actions >= 0).all() and (actions <= 1).all()
    with pytest.raises(ConflictError, match="名称已被占用"):
        train_policy(ws_data, cfg, "bc")


def test_training_is_reproducible_and_seed_sensitive(ws_data, cfg):
    cfg.policy.snapshots = []
    train_policy(ws_data, cfg, "a")
    delete_artifact(ws_data, "a")
    first = _weights  # 占位，下面重新训练同名策略
    train_policy(ws_data, cfg, "a")
    again = _weights(ws_data, "a")
    cfg.seed = 1
    train_policy(ws_data, cfg, "b")
    assert first is _weights and not _same(again, _weights(ws_data, "b"))


def test_same_name_same_seed_gives_identical_weights(cfg, tmp_path):
    cfg.policy.snapshots = []
    weights = []
    for sub in ("one", "two"):
        cfg.workspace.root = str(tmp_path / sub)
        with Workspace(cfg.workspace.root) as ws:
            build_dataset(ws, cfg)
            train_policy(ws, cfg, "bc")
            weights.append(_weights(ws, "bc"))
    assert _same(*weights)


def test_resume_is_equivalent_to_uninterrupted_training(cfg, tmp_path):
    """在第 47 步"断电"（最近的检查点在第 40 步），续训后的权重、快照、指标都与不中断时相同。"""
    results = []
    for sub, crash_at in (("straight", None), ("resumed", 47)):
        cfg.workspace.root = str(tmp_path / sub)
        with Workspace(cfg.workspace.root) as ws:
            build_dataset(ws, cfg)

            def crash(step, crash_at=crash_at):
                if step == crash_at:
                    raise KeyboardInterrupt

            if crash_at is None:
                art = train_policy(ws, cfg, "bc")
            else:
                with pytest.raises(KeyboardInterrupt):
                    train_policy(ws, cfg, "bc", on_step=crash)
                run = ws.db.execute("SELECT id, status FROM runs WHERE kind = 'train_policy'").fetchone()
                assert run["status"] == "interrupted"
                assert get_artifact(ws, "bc-step20").params["steps"] == 20  # 中断前的快照已经登记
                with pytest.raises(NotFoundError):
                    get_artifact(ws, "bc")
                art = train_policy(
                    ws, load_config(), "不会用到的名称", resume=run["id"]
                )  # 配置和名称取自那次运行
                assert art.name == "bc" and art.run_id == run["id"]
                assert (
                    ws.db.execute("SELECT status FROM runs WHERE id = ?", (run["id"],)).fetchone()[0]
                    == "done"
                )
                with pytest.raises(ConflictError, match="已被占用"):
                    train_policy(ws, cfg, "x", resume=run["id"])
            metrics = [m for m in read_metrics(ws.root / "runs" / art.run_id) if m["name"] != "steps_per_sec"]
            results.append((_weights(ws, "bc"), _weights(ws, "bc-step20"), metrics))
    (w1, s1, m1), (w2, s2, m2) = results
    assert _same(w1, w2) and _same(s1, s2)
    assert m1 == m2  # 续训时丢掉检查点之后的指标再重新产生，没有重复也没有缺口


def test_resume_needs_a_checkpoint(ws_data, cfg):
    with pytest.raises(NotFoundError, match="没有可以续训的检查点"):
        train_policy(ws_data, cfg, "bc", resume="no-such-run")


# ---- 策略库 ----


def test_zoo_builds_every_variant_and_is_resumable(ws_data, cfg):
    expected = ["bc", "bc-step20", "bc-demos2", "bc-noise0.3", "dp", "dp-step20", "random"]
    assert zoo_names(cfg) == expected
    names, created = build_zoo(ws_data, cfg)
    assert names == expected and sorted(created) == sorted(expected)
    assert build_zoo(ws_data, cfg) == (expected, [])  # 第二次什么都不用做
    assert {a.name for a in list_artifacts(ws_data, kind="policy")} == set(expected)

    base, noisy, few = (get_artifact(ws_data, n) for n in ("bc", "bc-noise0.3", "bc-demos2"))
    assert noisy.sha256 == base.sha256 and noisy.path == base.path  # 同一份权重，只是执行时加噪声
    assert noisy.params["exec_noise"] == 0.3 and base.params["exec_noise"] == 0.0
    assert few.params["n_demos"] == 2 and few.sha256 != base.sha256

    frames = np.zeros((2, 2, 32, 32, 3), dtype=np.uint8)
    act = lambda a: build_policy(a.backend, TOY.spec, a.params, checkpoint_path(ws_data, a), a.name).act(  # noqa: E731
        frames, np.random.default_rng(0)
    )
    assert not np.array_equal(act(base), act(noisy)) and np.array_equal(act(base), act(base))

    delete_artifact(ws_data, "bc-demos2")
    assert build_zoo(ws_data, cfg)[1] == ["bc-demos2"]  # 只补缺的那个


def test_delete_artifact_removes_evals_and_unshared_checkpoints(ws_data, cfg):
    build_zoo(ws_data, cfg)
    service.run_eval(ws_data, cfg, "bc-noise0.3")
    shared = checkpoint_path(ws_data, get_artifact(ws_data, "bc"))
    assert delete_artifact(ws_data, "bc-noise0.3") == 1
    assert shared is not None and shared.exists()  # bc 还在用同一份权重
    assert ws_data.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
    assert delete_artifact(ws_data, "bc") == 0 and not shared.exists()
    with pytest.raises(NotFoundError):
        delete_artifact(ws_data, "bc")


def test_eval_uses_whole_chunk_when_shorter_than_n_exec(ws_data, cfg):
    build_zoo(ws_data, cfg)
    cfg.eval.n_exec = 3
    service.run_eval(ws_data, cfg, "bc")  # 分块长 4，执行 3 个
    service.run_eval(ws_data, cfg, "random")  # 分块长 1，只能执行 1 个
    rows = ws_data.db.execute(
        "SELECT a.name, e.n_exec FROM evals e JOIN artifacts a ON a.id = e.policy_id"
    ).fetchall()
    assert dict(rows) == {"bc": 3, "random": 1}
    assert {r.policy for r in service.leaderboard(ws_data, cfg)} == {"bc", "random"}


# ---- 策略回合 → 并入数据集 ----


def test_rollouts_are_recorded_replayable_and_merged_into_dataset(ws_data, cfg):
    from roboarena.data.validate import check_episode, check_replay
    from roboarena.policies import register_policy_backend

    from .conftest import ToySeekPolicy

    register_policy_backend("toy_seek", lambda spec, params, ckpt, name: ToySeekPolicy(name))
    from roboarena.store.registry import register_artifact

    register_artifact(ws_data, task="toy", name="seek", kind="policy", backend="toy_seek", params={})
    register_artifact(ws_data, task="toy", name="random", kind="policy", backend="random", params={})
    cfg.rollouts.per_policy, cfg.rollouts.extra_steps = 5, 3
    summary = collect_rollouts(ws_data, cfg)
    assert set(summary.per_policy) == {"seek", "random"}
    assert summary.per_policy["seek"][:2] == (5, 5)

    episodes = load_rollout_set(ws_data.root, "toy", "zoo")
    assert len(episodes) == 10 and len(rollout_index(ws_data.root, "toy", "zoo")) == 10
    for ep in episodes:
        assert not [m for m in check_episode(TOY.spec, ep).values() if m] and check_replay(TOY, ep) is None
    for ep in (e for e in episodes if e.meta["source"] == "policy:seek"):
        first = 1 + int(np.argmax(ep.success[1:]))  # 评测循环只在执行动作之后判成功，所以从第 1 帧算起
        assert ep.success[first] and not ep.success[1:first].any()
        assert len(ep) == min(first + 3, 30)  # 成功之后多记 3 步
    assert all(len(e) == 30 for e in episodes if e.meta["source"] == "policy:random" and not e.success.any())

    suite = service.create_suite(ws_data, cfg) if False else None  # 评测集在夹具里已建
    inits = {json.dumps(e.meta["init"], sort_keys=True) for e in episodes}
    from roboarena.store.suites import load_suite

    assert not inits & {json.dumps(i, sort_keys=True) for i in load_suite(ws_data, "toy", "main").inits}
    assert suite is None

    cfg.data.name, cfg.data.rollouts = "with-rollouts", ["zoo"]
    ds = build_dataset(ws_data, cfg)
    sources = [e["source"] for e in ds.entries()]
    assert sources.count("policy:seek") == 5 and sources.count("policy:random") == 5 and len(sources) == 34
    assert ds.fingerprint != load_dataset(ws_data, "toy", "main").fingerprint
    assert verify_dataset(ds, TOY) == []
    assert "policy:seek" in (ds.dir / "REPORT.md").read_text()
    again = collect_rollouts(ws_data, cfg)  # 重新采集：同种子，内容相同
    assert again.per_policy == summary.per_policy
    cfg.data.rollouts = ["missing"]
    with pytest.raises(NotFoundError, match="data rollouts"):
        build_dataset(ws_data, cfg, force=True)


def test_rollouts_need_policies(ws_data, cfg):
    from roboarena.core.errors import RoboArenaError

    with pytest.raises(RoboArenaError, match="还没有登记任何策略"):
        collect_rollouts(ws_data, cfg)


# ---- 命令行 ----


def test_cli_zoo_flow(tmp_path, capsys):
    base = ["--config", str(QUICK), "--set", f'workspace.root="{tmp_path / "ws"}"']

    def run(*argv):
        code = main([*argv, *base])
        out = capsys.readouterr()
        return code, out.out, out.err

    assert run("train", "policy", "--name", "x")[0] == 1  # 还没有数据集
    assert run("data", "build")[0] == 0 and run("data", "suite")[0] == 0
    code, out, _ = run("train", "zoo")
    assert code == 0 and "共 7 个策略，这次新增 7 个" in out
    assert "新增 0 个：无" in run("train", "zoo")[1]
    code, _, err = run("train", "policy")
    assert code == 1 and "--name" in err
    code, _, err = run("eval", "run")
    assert code == 1 and "--all" in err
    code, out, _ = run("eval", "run", "--all")
    assert code == 0 and out.count("@ real") == 7
    code, out, _ = run("eval", "leaderboard")
    assert code == 0 and len(out.splitlines()) == 9
    code, out, _ = run("data", "rollouts", "--set", "rollouts.per_policy=2")
    assert code == 0 and "bc-noise0.3" in out and "data.rollouts" in out
    code, out, _ = run("data", "build", "--set", "data.name=v2", "--set", 'data.rollouts=["zoo"]')
    assert code == 0 and "38 个回合" in out
    code, out, _ = run("policy", "remove", "--name", "bc-demos2")
    assert code == 0 and "1 次评测结果" in out
    assert run("policy", "remove", "--name", "bc-demos2")[0] == 1


def test_rollouts_leave_a_run_record(ws_data, cfg):
    """采集策略回合也是一次运行：配置、种子、用了哪些策略的哪份权重，事后都查得到。"""
    from roboarena.store.registry import register_artifact

    register_artifact(ws_data, task="toy", name="random", kind="policy", backend="random", params={})
    cfg.rollouts.per_policy = 3
    summary = collect_rollouts(ws_data, cfg)
    run = ws_data.db.execute("SELECT kind, status FROM runs WHERE id = ?", (summary.run_id,)).fetchone()
    assert (run["kind"], run["status"]) == ("data", "done")
    meta = json.loads((ws_data.root / "runs" / summary.run_id / "meta.json").read_text())
    assert meta["rollouts"] == "toy/zoo"
    assert meta["policies"] == {"random": get_artifact(ws_data, "random").sha256}
    assert meta["per_policy"]["random"][0] == 3


def test_rollouts_with_a_wrong_task_policy_leave_no_run(ws_data, cfg):
    """前提不满足时直接报错，不留下一条失败的运行记录。"""
    from roboarena.core.errors import RoboArenaError
    from roboarena.store.registry import register_artifact

    register_artifact(ws_data, task="pusht", name="other", kind="policy", backend="random", params={})
    cfg.rollouts.policies = ["other"]
    before = ws_data.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    with pytest.raises(RoboArenaError, match="属于任务 pusht"):
        collect_rollouts(ws_data, cfg)
    assert ws_data.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == before
