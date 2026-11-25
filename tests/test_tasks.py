"""任务契约测试：每个任务插件都要通过同一组检查。新增任务时把名称加进 TASKS。"""

import json

import numpy as np
import pytest

from roboarena.core.errors import NotFoundError
from roboarena.tasks import get_task, list_tasks
from roboarena.tasks.toy import SUCCESS_RADIUS, ToyInit, ToySim

TASKS = ["toy"]


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
    frames = sim.reset(task.sample_inits(n, seed=0))
    assert frames.shape == (n, *spec.frame_shape) and frames.dtype == np.uint8
    result = sim.step(_random_actions(spec, np.random.default_rng(0), n))
    assert result.frames.shape == (n, *spec.frame_shape) and result.frames.dtype == np.uint8
    assert result.progress.shape == (n,) and result.progress.dtype == np.float32
    assert ((result.progress >= 0) & (result.progress <= 1)).all()
    assert result.success.shape == (n,) and result.success.dtype == bool
    assert set(result.labels) == set(spec.label_fields)
    for name, dim in spec.label_fields.items():
        assert result.labels[name].shape == (n, dim)
    sim.close()


def test_real_sim_is_deterministic_and_batch_independent(task):
    spec, n, steps = task.spec, 4, 6
    inits = task.sample_inits(n, seed=3)
    actions = [_random_actions(spec, np.random.default_rng(t), n) for t in range(steps)]

    def rollout(indices):
        sim = task.make_real_sim()
        frames = [sim.reset([inits[i] for i in indices])]
        progress = []
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
    frame = sim.reset([ToyInit(agent=(0.25, 0.75), goal=(0.75, 0.25))])[0]
    assert tuple(frame[24, 8]) == (220, 0, 0)  # 点：x = 0.25 → 第 8 列，y = 0.75 → 第 24 行
    assert tuple(frame[8, 24]) == (0, 170, 0)
    assert tuple(frame[0, 0]) == (255, 255, 255)
