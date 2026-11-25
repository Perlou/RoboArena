"""数据集：一组回合文件加一份清单。

    datasets/<任务>/<名称>/
        episodes/episode-000000.npz …
        manifest.json     回合清单（文件、来源、长度、摘要、所属的组）、切分、指纹、归一化统计量
        REPORT.md         校验报告

切分以"组"为单位：一条演示和由它加噪重放出来的回合属于同一组，整组进同一个子集，
这样验证集里不会出现和训练集共用同一条动作序列的回合。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboarena.config import Config
from roboarena.core.errors import ConfigError, ConflictError, IntegrityError, NotFoundError
from roboarena.core.hashing import combine_digests
from roboarena.core.log import get_logger
from roboarena.core.seeding import derive_seed, make_rng
from roboarena.data.collect import collect, noisy_replay_actions, random_walk_actions
from roboarena.data.episodes import episode_digest, load_episode, save_episode
from roboarena.data.rollouts import load_rollout_set
from roboarena.data.validate import CHECKS, check_episode, check_replay
from roboarena.store.runs import start_run
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task
from roboarena.tasks.api import Episode, Task

log = get_logger(__name__)

SPLITS = ("train", "val", "test")
N_REPLAY_CHECKS = 8  # 每种采集来源大约抽多少个回合做重放核对（等间隔抽）


@dataclass(frozen=True)
class Dataset:
    dir: Path
    manifest: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]

    def entries(self, split: str | None = None) -> list[dict[str, Any]]:
        if split is None:
            return self.manifest["episodes"]
        return [self.manifest["episodes"][i] for i in self.manifest["splits"][split]]

    def episodes(self, split: str | None = None) -> Iterator[Episode]:
        for entry in self.entries(split):
            yield load_episode(self.dir / entry["file"])


def dataset_dir(ws: Workspace, task: str, name: str) -> Path:
    return ws.root / "datasets" / task / name


def load_dataset(ws: Workspace, task: str, name: str) -> Dataset:
    path = dataset_dir(ws, task, name) / "manifest.json"
    if not path.exists():
        raise NotFoundError(f"任务 {task} 没有名为 {name} 的数据集，先运行 roboarena data build")
    return Dataset(path.parent, json.loads(path.read_text(encoding="utf-8")))


def split_groups(
    groups: Sequence[str], val_frac: float, test_frac: float, rng: np.random.Generator
) -> dict[str, str]:
    """把组分到 train / val / test。演示组和其他组分开洗牌，保证每个子集里两类数据都有。"""
    assignment: dict[str, str] = {}
    unique = sorted(set(groups))
    for kind in (True, False):
        part = [g for g in unique if g.startswith("demo-") == kind]
        order = rng.permutation(len(part))
        n_test, n_val = round(len(part) * test_frac), round(len(part) * val_frac)
        for rank, index in enumerate(order):
            assignment[part[index]] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    return assignment


class _Moments:
    """流式累计均值与标准差，忽略 NaN。"""

    def __init__(self) -> None:
        self.n: np.ndarray | float = 0.0
        self.s: np.ndarray | float = 0.0
        self.ss: np.ndarray | float = 0.0

    def add(self, x: np.ndarray) -> None:
        x = x.reshape(len(x), -1).astype(np.float64)
        ok = ~np.isnan(x)
        self.n = self.n + ok.sum(axis=0)
        self.s = self.s + np.where(ok, x, 0.0).sum(axis=0)
        self.ss = self.ss + np.where(ok, x * x, 0.0).sum(axis=0)

    def result(self) -> dict[str, list[float]] | None:
        n = np.asarray(self.n, dtype=np.float64)
        if n.ndim == 0 or (n == 0).any():
            return None
        mean = self.s / n
        std = np.sqrt(np.maximum(self.ss / n - mean * mean, 0.0))
        return {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()}


def _plan_sources(task: Task, cfg: Config, cache: Path) -> Iterator[tuple[str, list[Episode]]]:
    """按来源依次产出回合。演示来自任务插件，其余在真实仿真器里采集。种子全部由（全局种子，数据集名，来源）派生。"""
    data, spec, name = cfg.data, task.spec, cfg.data.name
    demos = [] if data.n_demos == 0 else task.demos(cache, None if data.n_demos < 0 else data.n_demos)
    for ep in demos:
        complete = not (np.isnan(ep.progress).any() or any(np.isnan(v).any() for v in ep.labels.values()))
        ep.meta.update(group=f"demo-{ep.meta['index']}", labeled=complete)
    if demos:
        yield "demo", demos

    if data.n_replay > 0:
        if not demos:
            raise ConfigError("data.n_replay > 0 需要演示数据作为动作来源，但 data.n_demos = 0")
        rng = make_rng(cfg.seed, f"data:{name}:replay")
        inits = task.sample_inits(data.n_replay, derive_seed(cfg.seed, f"data:{name}:replay:inits"))
        seqs, metas = [], []
        for i in range(data.n_replay):
            demo = demos[i % len(demos)]
            noise = float(rng.uniform(0.0, 2.0 * data.replay_noise))  # 每个回合的噪声强度不同
            seqs.append(noisy_replay_actions(spec, demo.actions, noise, rng))
            metas.append(
                {"source": "replay", "index": i, "group": demo.meta["group"], "noise": noise, "labeled": True}
            )
        yield "replay", collect(task, inits, seqs, metas, data.batch)

    if data.n_random_walk > 0:
        rng = make_rng(cfg.seed, f"data:{name}:random_walk")
        inits = task.sample_inits(data.n_random_walk, derive_seed(cfg.seed, f"data:{name}:random_walk:inits"))
        seqs = [random_walk_actions(spec, data.random_walk_steps, rng) for _ in range(data.n_random_walk)]
        metas = [
            {"source": "random_walk", "index": i, "group": f"walk-{i}", "labeled": True}
            for i in range(data.n_random_walk)
        ]
        yield "random_walk", collect(task, inits, seqs, metas, data.batch)

    for set_name in data.rollouts:
        yield f"rollouts:{set_name}", load_rollout_set(cache.parent, spec.name, set_name)


def build_dataset(ws: Workspace, cfg: Config, *, force: bool = False) -> Dataset:
    """按配置生成数据集。任何一项校验不通过都会报错并且不留下清单。"""
    task = get_task(cfg.task.name)
    spec = task.spec
    out = dataset_dir(ws, spec.name, cfg.data.name)
    if out.exists():
        if not force:
            raise ConflictError(f"数据集 {spec.name}/{cfg.data.name} 已存在；要重建请加 --force")
        shutil.rmtree(out)
    (out / "episodes").mkdir(parents=True)

    with start_run(ws, "data", config=cfg.to_dict(), config_toml=cfg.to_toml(), seed=cfg.seed) as run:
        entries: list[dict[str, Any]] = []
        moments: list[dict[str, _Moments]] = []
        problems: list[str] = []
        replay_checked: dict[str, int] = {}
        for source, episodes in _plan_sources(task, cfg, ws.root / "cache"):
            for n, ep in enumerate(episodes):
                file = f"episodes/episode-{len(entries):06d}.npz"
                failed = {k: v for k, v in check_episode(spec, ep).items() if v is not None}
                if not failed and source != "demo" and n % max(len(episodes) // N_REPLAY_CHECKS, 1) == 0:
                    replay_checked[source] = replay_checked.get(source, 0) + 1
                    if (message := check_replay(task, ep)) is not None:
                        failed["可重放"] = message
                problems += [f"{file}（{source}）{check}：{message}" for check, message in failed.items()]
                if failed:
                    continue
                save_episode(out / file, ep)
                stats = {"actions": _Moments(), **{f"labels.{k}": _Moments() for k in ep.labels}}
                stats["actions"].add(ep.actions)
                for key, value in ep.labels.items():
                    stats[f"labels.{key}"].add(value)
                moments.append(stats)
                entries.append(
                    {
                        "file": file,
                        "source": ep.meta["source"],
                        "group": ep.meta["group"],
                        "length": len(ep),
                        "labeled": bool(ep.meta["labeled"]),
                        "succeeded": bool(ep.success.any()),
                        "init": ep.meta.get("init"),
                        "digest": episode_digest(ep),
                    }
                )
            log.info("来源处理完成", extra={"source": source, "episodes": len(episodes)})

        digests = [e["digest"] for e in entries]
        if len(set(digests)) != len(digests):
            problems.append("有内容完全相同的回合")
        if problems:
            shutil.rmtree(out)
            raise IntegrityError("数据校验未通过：\n  " + "\n  ".join(problems[:20]))

        assignment = split_groups(
            [e["group"] for e in entries],
            cfg.data.val_frac,
            cfg.data.test_frac,
            make_rng(cfg.seed, f"data:{cfg.data.name}:split"),
        )
        splits = {s: [i for i, e in enumerate(entries) if assignment[e["group"]] == s] for s in SPLITS}
        total = {key: _Moments() for key in moments[0]}
        for i in splits["train"]:  # 归一化统计量只用训练部分
            for key, m in moments[i].items():
                total[key].n, total[key].s, total[key].ss = (
                    total[key].n + m.n,
                    total[key].s + m.s,
                    total[key].ss + m.ss,
                )

        manifest = {
            "task": spec.name,
            "name": cfg.data.name,
            "fingerprint": combine_digests(digests),
            "run_id": run.id,
            "n_episodes": len(entries),
            "n_frames": sum(e["length"] + 1 for e in entries),
            "splits": splits,
            "stats": {key: m.result() for key, m in total.items()},
            "replay_checked": replay_checked,
            "episodes": entries,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        dataset = Dataset(out, manifest)
        (out / "REPORT.md").write_text(render_report(dataset), encoding="utf-8")
        run.update_meta(dataset_fp=dataset.fingerprint, dataset=f"{spec.name}/{cfg.data.name}")
        ws.db.execute("UPDATE runs SET dataset_fp = ? WHERE id = ?", (dataset.fingerprint, run.id))
        ws.db.commit()
    return dataset


def verify_dataset(dataset: Dataset, task: Task) -> list[str]:
    """重新读取每个回合文件：核对摘要与指纹、重跑逐回合检查、检查切分没有把同一组拆开。返回问题列表。"""
    problems: list[str] = []
    digests = []
    for entry in dataset.entries():
        path = dataset.dir / entry["file"]
        if not path.exists():
            problems.append(f"{entry['file']}：文件不见了")
            continue
        ep = load_episode(path)
        digests.append(episode_digest(ep))
        if digests[-1] != entry["digest"]:
            problems.append(f"{entry['file']}：内容与清单里的摘要不一致")
        problems += [
            f"{entry['file']} {k}：{v}" for k, v in check_episode(task.spec, ep).items() if v is not None
        ]
    if combine_digests(digests) != dataset.fingerprint:
        problems.append("数据集指纹与清单不一致")
    where: dict[str, str] = {}
    listed = sorted(i for s in SPLITS for i in dataset.manifest["splits"][s])
    if listed != list(range(len(dataset.entries()))):
        problems.append("切分没有恰好覆盖每个回合一次")
    for split in SPLITS:
        for entry in dataset.entries(split):
            if where.setdefault(entry["group"], split) != split:
                problems.append(f"组 {entry['group']} 同时出现在 {where[entry['group']]} 和 {split}")
    return problems


def render_report(dataset: Dataset) -> str:
    m = dataset.manifest
    entries = m["episodes"]
    lines = [
        f"# 数据集 {m['task']}/{m['name']}",
        "",
        f"- 指纹：`{m['fingerprint']}`",
        f"- 生成它的运行：`{m['run_id']}`",
        f"- 回合：{m['n_episodes']}，画面：{m['n_frames']} 帧",
        "",
        "## 来源",
        "",
        "| 来源 | 回合 | 帧 | 带标签 | 回合内出现过成功 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in sorted({e["source"] for e in entries}):
        part = [e for e in entries if e["source"] == source]
        lines.append(
            f"| {source} | {len(part)} | {sum(e['length'] + 1 for e in part)} |"
            f" {sum(e['labeled'] for e in part)} | {sum(e['succeeded'] for e in part)} |"
        )
    lines += [
        "",
        "## 切分（按组，同一组不跨子集）",
        "",
        "| 子集 | 回合 | 帧 | 组 |",
        "| --- | --- | --- | --- |",
    ]
    for split in SPLITS:
        part = dataset.entries(split)
        lines.append(
            f"| {split} | {len(part)} | {sum(e['length'] + 1 for e in part)} | {len({e['group'] for e in part})} |"
        )
    lines += ["", "## 校验", "", "生成时每个回合都通过了以下检查，否则不会写出清单：", ""]
    lines += [f"- {name}：{check.__doc__}" for name, check in CHECKS.items()]
    checked = "、".join(f"{k} {v} 个" for k, v in m["replay_checked"].items()) or "无"
    lines += [
        f"- 可重放：从记录的初始状态重新执行动作，画面逐像素相同（抽查：{checked}）",
        "- 无重复：没有内容完全相同的回合",
        "",
        "## 归一化统计量（只用训练部分）",
        "",
        "| 量 | 均值 | 标准差 |",
        "| --- | --- | --- |",
    ]
    for key, stat in m["stats"].items():
        if stat is None:
            lines.append(f"| {key} | 训练部分没有这个量 | |")
        else:
            fmt = lambda xs: "[" + ", ".join(f"{x:.4g}" for x in xs) + "]"  # noqa: E731
            lines.append(f"| {key} | {fmt(stat['mean'])} | {fmt(stat['std'])} |")
    return "\n".join(lines) + "\n"
