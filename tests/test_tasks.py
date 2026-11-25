"""任务契约测试：每个任务插件都要通过同一组检查。新增任务时把名称加进 TASKS。"""

import json

import numpy as np
import pytest

from roboarena.core.errors import NotFoundError
from roboarena.tasks import get_task, list_tasks
from roboarena.tasks.toy import SUCCESS_RADIUS, ToyInit, ToySim

TASKS = ["toy", "pusht"]


def test_registry():
    assert set(TASKS) <= set(list_tasks())
    with pytest.raises(NotFoundError, match="已注册"):
        get_task("no-such-task")


@pytest.fixture(params=TASKS)
def task(request):
    return get_task(request.param)


def _random_actions(spec, rng, n):
    return rng.uniform(spec.action_low, spec.action_high, size=(n, spec.action_dim)).astype(np.float32)


def test_spec_is_consistent(task):
    spec = task.spec
    assert spec.frame_shape[2] == 3
    assert spec.action_low.shape == spec.action_high.shape == (spec.action_dim,)
    assert spec.action_low.dtype == spec.action_high.dtype == np.float32
    assert (spec.action_low < spec.action_high).all()
    assert spec.max_steps > 0 and spec.control_hz > 0
    json.dumps(spec.to_dict())  # 规格要能通过接口发给前端


def test_inits_are_seeded_and_serializable(task):
    a, b, c = task.sample_inits(5, seed=1), task.sample_inits(5, seed=1), task.sample_inits(5, seed=2)
    assert a == b and a != c
    for init in a:
        data = task.encode_init(init)
        assert task.decode_init(json.loads(json.dumps(data))) == init


def test_real_sim_matches_spec(task):
    spec, n = task.spec, 3
    sim = task.make_real_sim()
    first = sim.reset(task.sample_inits(n, seed=0))
    step = sim.step(_random_actions(spec, np.random.default_rng(0), n))
    for result in (first, step):  # 重置和步进返回同样完整的观测
        _check_result(result, spec, n)
    sim.close()


def _check_result(result, spec, n):
    assert result.frames.shape == (n, *spec.frame_shape) and result.frames.dtype == np.uint8
    assert result.progress.shape == (n,) and result.progress.dtype == np.float32
    assert ((result.progress >= 0) & (result.progress <= 1)).all()
    assert result.success.shape == (n,) and result.success.dtype == bool
    assert set(result.labels) == set(spec.label_fields)
    for name, dim in spec.label_fields.items():
        assert result.labels[name].shape == (n, dim) and result.labels[name].dtype == np.float32


def test_real_sim_is_deterministic_and_batch_independent(task):
    spec, n, steps = task.spec, 4, 6
    inits = task.sample_inits(n, seed=3)
    actions = [_random_actions(spec, np.random.default_rng(t), n) for t in range(steps)]

    def rollout(indices):
        sim = task.make_real_sim()
        first = sim.reset([inits[i] for i in indices])
        frames, progress = [first.frames], [first.progress]
        for a in actions:
            result = sim.step(a[indices])
            frames.append(result.frames)
            progress.append(result.progress)
        sim.close()
        return np.stack(frames), np.stack(progress)

    all_frames, all_progress = rollout(list(range(n)))
    again_frames, again_progress = rollout(list(range(n)))
    assert np.array_equal(all_frames, again_frames) and np.array_equal(all_progress, again_progress)
    for i in range(n):  # 一批一起跑和单独跑，结果相同
        one_frames, one_progress = rollout([i])
        assert np.array_equal(one_frames[:, 0], all_frames[:, i])
        assert np.allclose(one_progress[:, 0], all_progress[:, i])


def test_toy_dynamics():
    sim = ToySim()
    sim.reset([ToyInit(agent=(0.2, 0.2), goal=(0.8, 0.2))])
    first = sim.step(np.array([[0.8, 0.2]], dtype=np.float32))
    assert np.allclose(sim.agent, [[0.35, 0.2]])  # 每步最多移动 0.15
    assert np.isclose(first.progress[0], 1 - 0.45) and not first.success[0]
    for _ in range(3):
        last = sim.step(np.array([[0.8, 0.2]], dtype=np.float32))
    assert np.allclose(sim.agent, [[0.8, 0.2]]) and last.success[0]
    assert last.progress[0] > 1 - SUCCESS_RADIUS


def test_toy_render_shows_goal_and_agent():
    sim = ToySim()
    frame = sim.reset([ToyInit(agent=(0.25, 0.75), goal=(0.75, 0.25))]).frames[0]
    assert tuple(frame[24, 8]) == (220, 0, 0)  # 点：x = 0.25 → 第 8 列，y = 0.75 → 第 24 行
    assert tuple(frame[8, 24]) == (0, 170, 0)
    assert tuple(frame[0, 0]) == (255, 255, 255)


def test_sim_can_be_reset_with_a_different_batch_size(task):
    sim = task.make_real_sim()
    inits = task.sample_inits(3, seed=5)
    big = sim.reset(inits)
    small = sim.reset(inits[1:2])
    assert small.frames.shape[0] == 1 and np.array_equal(small.frames[0], big.frames[1])
    assert sim.step(_random_actions(task.spec, np.random.default_rng(0), 1)).frames.shape[0] == 1
    sim.close()


# ---- PushT 专有的事实（需求文档第 3 节）----


def test_pusht_spec_facts():
    spec = get_task("pusht").spec
    assert spec.frame_shape == (96, 96, 3) and spec.control_hz == 10 and spec.max_steps == 300
    assert spec.action_low.tolist() == [0, 0] and spec.action_high.tolist() == [512, 512]


def test_pusht_inits_follow_the_env_reset_distribution():
    states = np.array([i.state for i in get_task("pusht").sample_inits(500, seed=0)])
    assert states[:, :2].min() >= 50 and states[:, :2].max() < 450
    assert states[:, 2:4].min() >= 100 and states[:, 2:4].max() < 400
    assert -np.pi <= states[:, 4].min() and states[:, 4].max() <= np.pi


def test_pusht_reset_reports_the_requested_state():
    from roboarena.tasks.pusht import PushTInit

    sim = get_task("pusht").make_real_sim()
    inits = [
        PushTInit((60.0, 60.0, 300.0, 300.0, 0.5)),
        PushTInit((440.0, 80.0, 150.0, 350.0, -2.0)),
    ]  # 推杆离块很远
    first = sim.reset(inits)
    assert np.allclose(first.labels["agent_pos"], [i.state[:2] for i in inits])
    assert np.allclose(first.labels["block_pose"][:, 2], [i.state[4] for i in inits], atol=1e-5)
    assert not first.success.any()
    # 推杆原地不动：块不动，进度不变
    still = sim.step(first.labels["agent_pos"])
    assert np.allclose(still.labels["block_pose"], first.labels["block_pose"], atol=1e-3)
    assert np.allclose(still.progress, first.progress, atol=1e-6)
    with pytest.raises(ValueError, match="形状"):
        sim.step(np.zeros((3, 2), dtype=np.float32))
    sim.close()


def test_pusht_progress_and_success_at_the_goal():
    from gym_pusht.envs.pusht import PushTEnv

    from roboarena.tasks.pusht import PushTInit
    from roboarena.tasks.pusht.sim import SUCCESS_THRESHOLD

    env = PushTEnv()
    env.reset()
    assert SUCCESS_THRESHOLD == env.success_threshold
    goal_x, goal_y, goal_angle = (float(v) for v in env.goal_pose)

    # gym-pusht 设定朝向时块的位置会偏移：先按目标位姿放一次，读回实际位置，再把偏差补回去
    sim = get_task("pusht").make_real_sim()
    naive = sim.reset([PushTInit((30.0, 30.0, goal_x, goal_y, goal_angle))])
    off_x, off_y = naive.labels["block_pose"][0, :2] - (goal_x, goal_y)
    assert naive.progress[0] < 0.5 and not naive.success[0]  # 没补偏差时只盖住一小半
    exact = sim.reset([PushTInit((30.0, 30.0, goal_x - off_x, goal_y - off_y, goal_angle))])
    assert np.allclose(exact.labels["block_pose"][0], (goal_x, goal_y, goal_angle), atol=1e-2)
    assert exact.progress[0] > 0.99 and exact.success[0]
    sim.close()
