import numpy as np
import pytest

from roboarena.core.errors import NotFoundError, RoboArenaError
from roboarena.eval import service
from roboarena.eval.runner import evaluate
from roboarena.eval.stats import wilson_interval
from roboarena.store.registry import register_artifact
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task
from roboarena.tasks.api import StepResult

from .conftest import ToySeekPolicy

# ---- Wilson 区间：对手算例子 ----


def test_wilson_hand_computed():
    # 5/10，z = 1.96：中心 0.5，半宽 = 1.96·sqrt(0.025 + 0.009604) / 1.38416 = 0.2634
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    # 0/10：下界 0，上界 = z² / (n + z²) = 3.8415 / 13.8415 = 0.2775
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    # 对称性：10/10 的区间是 0/10 的镜像
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and low == pytest.approx(1 - 0.2775, abs=1e-4)


def test_wilson_edge_cases():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    narrow, wide = wilson_interval(50, 100), wilson_interval(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]  # 样本越多区间越窄
    loose = wilson_interval(5, 10, confidence=0.5)
    assert loose[1] - loose[0] < wide[1] - wide[0]
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


# ---- 评测循环：用可以预先算出结果的假仿真器与假策略 ----


class CountingSim:
    """第 i 个回合在第 succeed_at[i] 步成功（0 表示永不成功）。画面的全部像素等于当前步数，进度 = 步数 / 10。"""

    name = "fake"

    def __init__(self):
        self.actions = []

    def reset(self, inits):
        self.succeed_at = np.array(inits)
        self.t = 0
        return np.zeros((len(inits), 2, 2, 3), dtype=np.uint8)

    def step(self, actions):
        self.t += 1
        self.actions.append(np.array(actions))
        n = len(self.succeed_at)
        return StepResult(
            frames=np.full((n, 2, 2, 3), self.t, dtype=np.uint8),
            progress=np.full(n, min(self.t / 10, 1.0), dtype=np.float32),
            success=self.succeed_at == self.t,
        )

    def close(self):
        pass


class ChunkPolicy:
    """输出的第 k 个动作等于 100·调用次数 + k，并记下每次看到的画面历史。"""

    name = "chunky"

    def __init__(self, obs_frames=2, chunk=3):
        self.obs_frames, self.chunk = obs_frames, chunk
        self.seen = []

    def act(self, frames, rng):
        self.seen.append(frames[0, :, 0, 0, 0].tolist())
        call = len(self.seen) - 1
        row = 100 * call + np.arange(self.chunk, dtype=np.float32)
        return np.tile(row[None, :, None], (len(frames), 1, 1))


def _eval(policy, sim, inits, max_steps=6, n_exec=2):
    return evaluate(policy, sim, inits, max_steps=max_steps, n_exec=n_exec, rng=np.random.default_rng(0))


def test_runner_records_success_step_and_best_progress():
    results = _eval(ChunkPolicy(), CountingSim(), [3, 0, 1])
    assert [(r.success, r.steps) for r in results] == [(True, 3), (False, 6), (True, 1)]
    # 成功之后不再更新最大进度：回合 0 停在第 3 步，回合 2 停在第 1 步
    assert [r.max_progress for r in results] == pytest.approx([0.3, 0.6, 0.1])


def test_runner_executes_n_exec_actions_per_call_and_rolls_history():
    policy, sim = ChunkPolicy(obs_frames=2, chunk=3), CountingSim()
    _eval(policy, sim, [0], max_steps=5, n_exec=2)
    assert [a[0, 0] for a in sim.actions] == [0, 1, 100, 101, 200]  # 每次只用分块的前 2 个；第 5 步后到上限
    assert policy.seen == [[0, 0], [1, 2], [3, 4]]  # 首帧重复填满历史，之后每步滚动一帧


def test_runner_stops_early_when_all_succeed():
    sim = CountingSim()
    _eval(ChunkPolicy(), sim, [1, 2], max_steps=50, n_exec=1)
    assert len(sim.actions) == 2


def test_runner_rejects_bad_n_exec_and_bad_policy_output():
    with pytest.raises(ValueError, match="n_exec"):
        _eval(ChunkPolicy(chunk=2), CountingSim(), [0], n_exec=3)

    class Broken(ChunkPolicy):
        def act(self, frames, rng):
            return np.zeros((len(frames), 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="形状"):
        _eval(Broken(chunk=3), CountingSim(), [0])


# ---- 玩具任务走完全流程：评测集 → 登记策略 → 评测 → 入库 → 排行榜 ----


def test_seek_policy_solves_toy_from_pixels():
    task = get_task("toy")
    inits = task.sample_inits(64, seed=0)
    results = evaluate(
        ToySeekPolicy(), task.make_real_sim(), inits, max_steps=30, n_exec=1, rng=np.random.default_rng(0)
    )
    assert all(r.success for r in results)
    assert max(r.steps for r in results) <= 8  # 最远约 1.13，每步 0.15


def _setup(cfg):
    ws = Workspace(cfg.workspace.root)
    service.create_suite(ws, cfg)
    register_artifact(ws, task="toy", name="seek", kind="policy", backend="toy_seek", params={})
    register_artifact(ws, task="toy", name="random", kind="policy", backend="random", params={})
    return ws


def test_toy_end_to_end(cfg):
    ws = _setup(cfg)
    seek = service.run_eval(ws, cfg, "seek")
    rand = service.run_eval(ws, cfg, "random")
    assert (seek.n, seek.successes, seek.skipped) == (cfg.eval.n_inits, cfg.eval.n_inits, False)
    assert rand.successes < seek.successes

    board = service.leaderboard(ws, cfg)
    assert [r.policy for r in board] == ["seek", "random"]
    assert board[0].rate == 1.0 and board[0].ci_high == 1.0 and board[0].ci_low < 1.0
    assert board[0].mean_max_progress > board[1].mean_max_progress
    assert board[1].successes == rand.successes

    runs = ws.db.execute("SELECT kind, status FROM runs").fetchall()
    assert [tuple(r) for r in runs] == [("eval", "done")] * 2
    ws.close()


def test_eval_is_idempotent_reproducible_and_keyed(cfg):
    ws = _setup(cfg)
    first = service.run_eval(ws, cfg, "random")
    again = service.run_eval(ws, cfg, "random")
    assert again.skipped and again.eval_id == first.eval_id and again.successes == first.successes
    assert ws.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1  # 跳过时不产生新的运行

    def episodes(eval_id):
        return [
            tuple(r)
            for r in ws.db.execute(
                "SELECT init_index, success, max_progress, steps FROM episodes WHERE eval_id = ?", (eval_id,)
            )
        ]

    before = episodes(first.eval_id)
    forced = service.run_eval(ws, cfg, "random", force=True)
    assert not forced.skipped and episodes(forced.eval_id) == before  # 同配置同种子，逐回合相同
    assert ws.db.execute("SELECT COUNT(*) FROM evals").fetchone()[0] == 1

    cfg.eval.seed = 1  # 换种子是另一次评测；排行榜把两次合并
    other = service.run_eval(ws, cfg, "random")
    assert not other.skipped and other.eval_id != forced.eval_id
    assert episodes(other.eval_id) != before
    assert service.leaderboard(ws, cfg)[0].n == 2 * cfg.eval.n_inits
    ws.close()


def test_eval_errors(cfg):
    ws = _setup(cfg)
    with pytest.raises(NotFoundError):
        service.run_eval(ws, cfg, "nobody")
    with pytest.raises(RoboArenaError, match="M3"):
        service.run_eval(ws, cfg, "seek", sim="wm-latent")
    register_artifact(ws, task="pusht", name="other-task", kind="policy", backend="random", params={})
    with pytest.raises(RoboArenaError, match="属于任务 pusht"):
        service.run_eval(ws, cfg, "other-task")
    cfg.eval.suite = "missing"
    with pytest.raises(NotFoundError, match="data suite"):
        service.run_eval(ws, cfg, "seek")
    ws.close()


def test_interrupted_eval_is_redone(cfg):
    ws = _setup(cfg)
    done = service.run_eval(ws, cfg, "seek")
    ws.db.execute("UPDATE evals SET status = 'running' WHERE id = ?", (done.eval_id,))
    ws.db.execute("DELETE FROM episodes WHERE eval_id = ?", (done.eval_id,))
    ws.db.commit()
    assert service.leaderboard(ws, cfg) == []  # 没跑完的评测不上榜
    redo = service.run_eval(ws, cfg, "seek")
    assert not redo.skipped and redo.successes == cfg.eval.n_inits
    ws.close()
