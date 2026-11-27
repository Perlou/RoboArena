"""用例（docs/PLAN.md 1.1 节）。单步的 toolcall 套件：一句指令 → 一次工具调用。

用例按「族」组织：一个族是一种意图，有若干条指令模板，模板里的槽位由种子确定的采样填入。
标准答案由程序给出。三个桶：seen（模板会进微调语料）、heldout（留出的模板，以及整族留出的族）、
handwritten（人工编写，见 handwritten.py）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from roboarena.aieval import fixture
from roboarena.aieval.handwritten import HANDWRITTEN
from roboarena.core.hashing import canonical_json, sha256_text
from roboarena.core.seeding import make_rng
from roboarena.llm.api import ToolCall

BUCKETS = ("seen", "heldout", "handwritten")


@dataclass(frozen=True)
class Case:
    id: str
    family: str  # 用例族：一种意图
    bucket: str  # seen / heldout / handwritten
    group: str  # 同一条种子指令的各种说法同组，切分以组为单位
    turns: tuple[str, ...]  # 用户的话；多轮用例有多条
    gold_calls: tuple[ToolCall, ...] | None  # 单步用例的标准工具调用
    check: str  # 检查器的名称


Slots = dict[str, Any]


@dataclass(frozen=True)
class Family:
    name: str
    tool: str
    sample: Callable[[np.random.Generator], Slots]  # 抽一组槽位值
    args: Callable[[Slots], dict[str, Any]]  # 槽位值 → 标准参数
    seen: tuple[str, ...]  # 进微调语料的模板
    heldout: tuple[str, ...]  # 留出的模板；整族留出的族，模板全在这里


def _pick(rng: np.random.Generator, options: Sequence[Any]) -> Any:
    return options[int(rng.integers(len(options)))]


def _two_policies(rng: np.random.Generator) -> tuple[str, str]:
    i, j = rng.choice(len(fixture.POLICIES), size=2, replace=False)
    return fixture.POLICIES[int(i)], fixture.POLICIES[int(j)]


_KINDS = {"data": "数据", "train_policy": "训练", "eval": "评测"}
_SECTIONS = {
    "policy": "策略训练",
    "eval": "评测",
    "data": "数据采集",
    "diffusion": "扩散策略",
    "zoo": "策略库",
}
_STEPS = (300, 500, 800, 1000, 1500, 2000, 3000, 5000)
_LRS = ("1e-3", "3e-4", "1e-4", "5e-4", "2e-4")
_BATCHES = (16, 32, 64, 128, 256)
_NEW_NAMES = ("nova", "orbit", "pilot", "ridge", "sable", "tango", "umbra", "vega", "wren", "zephyr")
_BACKENDS = {"bc": "行为克隆", "diffusion": "扩散策略"}


def _run_id(rng: np.random.Generator) -> str:
    """形状与真实运行编号相同的编号。单步用例只看调用写得对不对，不要求这次运行存在。"""
    kind = _pick(rng, tuple(_KINDS))
    day, hour, minute, second = (int(rng.integers(n)) for n in (28, 24, 60, 60))
    tail = "".join(f"{int(b):02x}" for b in rng.integers(256, size=4))
    return f"202611{day + 1:02d}-{hour:02d}{minute:02d}{second:02d}-{kind}-{tail}"


def _sample_runs(rng: np.random.Generator) -> Slots:
    kind = _pick(rng, tuple(_KINDS))
    return {"kind": kind, "kind_zh": _KINDS[kind], "n": int(_pick(rng, (3, 5, 8, 15, 20)))}


def _sample_section(rng: np.random.Generator) -> Slots:
    section = _pick(rng, tuple(_SECTIONS))
    return {"section": section, "section_zh": _SECTIONS[section]}


def _sample_tuning(rng: np.random.Generator) -> Slots:
    backend = _pick(rng, tuple(_BACKENDS))
    return {
        "name": _pick(rng, _NEW_NAMES),
        "backend": backend,
        "backend_zh": _BACKENDS[backend],
        "section": "policy" if backend == "bc" else "diffusion",
        "steps": int(_pick(rng, _STEPS)),
        "lr": _pick(rng, _LRS),
        "batch": int(_pick(rng, _BATCHES)),
    }


def _sample_compare(rng: np.random.Generator) -> Slots:
    a, b = _two_policies(rng)
    return {"a": a, "b": b, "suite": _pick(rng, fixture.SUITES)}


FAMILIES: tuple[Family, ...] = (
    Family(
        "list_policies",
        "list_policies",
        lambda rng: {},
        lambda s: {},
        seen=("现在登记了哪些策略？", "把所有策略列出来", "注册表里都有什么策略", "列一下当前任务的策略"),
        heldout=("我们手上一共有哪几个策略可以用？", "策略库里现在都有谁"),
    ),
    Family(
        "leaderboard",
        "leaderboard",
        lambda rng: {"suite": _pick(rng, fixture.SUITES)},
        lambda s: {"suite": s["suite"]},
        seen=(
            "看一下评测集 {suite} 的排行榜",
            "{suite} 上各个策略的成功率是多少",
            "给我 {suite} 这个评测集的排名",
            "评测集 {suite} 上谁的成功率最高？把排行榜调出来",
        ),
        heldout=("{suite} 那套评测跑下来，各家策略表现怎么样", "我想知道在 {suite} 上策略们的名次"),
    ),
    Family(
        "leaderboard_default",
        "leaderboard",
        lambda rng: {},
        lambda s: {},
        seen=("看一下排行榜", "现在的排行榜是什么样", "各个策略的成功率排名", "把当前评测集的排行榜给我"),
        heldout=("目前哪个策略最强？看下榜单", "策略的名次现在怎么排的"),
    ),
    Family(  # 整族留出：工具见过，top 这个参数没见过
        "leaderboard_top",
        "leaderboard",
        lambda rng: {"suite": _pick(rng, fixture.SUITES), "k": int(_pick(rng, (1, 2, 3, 5)))},
        lambda s: {"suite": s["suite"], "top": s["k"]},
        seen=(),
        heldout=(
            "评测集 {suite} 上的前 {k} 名是谁",
            "只看 {suite} 排行榜的前 {k} 个",
            "{suite} 上成功率最高的 {k} 个策略",
            "给我 {suite} 的榜单，取前 {k} 名就行",
            "列出在 {suite} 上排名前 {k} 的策略",
        ),
    ),
    Family(
        "compare",
        "compare_policies",
        _sample_compare,
        lambda s: {"a": s["a"], "b": s["b"], "suite": s["suite"]},
        seen=(
            "比较 {a} 和 {b} 在评测集 {suite} 上的表现",
            "在 {suite} 上，{a} 和 {b} 哪个更好？",
            "{a} 对比 {b}，评测集用 {suite}",
            "{suite} 上 {a} 与 {b} 的差别显著吗",
        ),
        heldout=("拿 {a} 跟 {b} 比一比，看 {suite} 上的结果", "{a} 是不是真的比 {b} 强？按 {suite} 的数据说"),
    ),
    Family(
        "list_runs",
        "list_runs",
        _sample_runs,
        lambda s: {"kind": s["kind"], "limit": s["n"]},
        seen=(
            "列出最近 {n} 条{kind_zh}运行",
            "最近的 {n} 次{kind_zh}运行记录",
            "看一下最新的 {n} 条{kind_zh}类型的运行",
            "把最近 {n} 条{kind_zh}的运行记录调出来",
        ),
        heldout=("最近跑过哪些{kind_zh}？给我 {n} 条", "查{kind_zh}运行的历史，要最新的 {n} 条"),
    ),
    Family(
        "get_run",
        "get_run",
        lambda rng: {"run_id": _run_id(rng)},
        lambda s: {"run_id": s["run_id"]},
        seen=(
            "看一下运行 {run_id} 的详情",
            "运行 {run_id} 的状态和指标",
            "{run_id} 这次运行跑得怎么样",
            "查一下 {run_id}",
        ),
        heldout=("{run_id} 跑完了吗？最后的指标是多少", "帮我翻一下编号 {run_id} 的那次运行"),
    ),
    Family(
        "dataset_info",
        "dataset_info",
        lambda rng: {"name": _pick(rng, fixture.DATASETS)},
        lambda s: {"name": s["name"]},
        seen=(
            "数据集 {name} 有多少个回合",
            "看一下数据集 {name} 的概况",
            "{name} 这个数据集的训练、验证、测试各有多少",
            "数据集 {name} 一共多少帧",
        ),
        heldout=("{name} 那份数据有多大", "给我讲讲数据集 {name} 是怎么切分的"),
    ),
    Family(
        "describe_config",
        "describe_config",
        _sample_section,
        lambda s: {"section": s["section"]},
        seen=(
            "{section} 段有哪些配置项",
            "列出 {section} 的全部配置项和默认值",
            "配置里 {section} 这一段都能调什么",
            "看一下 {section} 段的配置说明",
        ),
        heldout=("{section} 相关的参数有哪些可以改", "我想了解 {section} 这段配置的每一项是什么意思"),
    ),
    Family(
        "check_overrides",
        "check_overrides",
        _sample_tuning,
        lambda s: {"overrides": [f"{s['section']}.steps={s['steps']}", f"{s['section']}.lr={s['lr']}"]},
        seen=(
            "检查一下这两个覆盖项对不对：{section}.steps={steps}、{section}.lr={lr}",
            "{section}.steps={steps} 和 {section}.lr={lr} 这样写合法吗",
            "帮我校验覆盖项 {section}.steps={steps} 与 {section}.lr={lr}",
            "把 {section} 段的 steps 设成 {steps}、lr 设成 {lr}，先检查能不能这样设，不要执行",
        ),
        heldout=(
            "我打算把 {section} 的训练步数改成 {steps}、学习率改成 {lr}，只帮我验一下这样配行不行，先别跑",
            "{section} 段：步数 {steps}，学习率 {lr}。这组设置合不合法？只校验",
        ),
    ),
    Family(
        "run_eval",
        "run_eval",
        lambda rng: {"policy": _pick(rng, fixture.POLICIES), "suite": _pick(rng, fixture.SUITES)},
        lambda s: {"policy": s["policy"], "suite": s["suite"]},
        seen=(
            "在评测集 {suite} 上评测策略 {policy}",
            "评测一下 {policy}，用 {suite}",
            "把 {policy} 放到 {suite} 上跑一遍评测",
            "用评测集 {suite} 给 {policy} 打个分",
        ),
        heldout=("{policy} 在 {suite} 上还没评过，现在跑一下", "跑评测：策略 {policy}，评测集 {suite}"),
    ),
    Family(
        "train_policy",
        "train_policy",
        _sample_tuning,
        lambda s: {
            "name": s["name"],
            "backend": s["backend"],
            "overrides": [f"{s['section']}.steps={s['steps']}", f"{s['section']}.batch={s['batch']}"],
        },
        seen=(
            "用{backend_zh}训练一个新策略，叫 {name}，训练 {steps} 步，批大小 {batch}",
            "训练策略 {name}：算法{backend_zh}，步数 {steps}，批大小 {batch}",
            "新训一个{backend_zh}策略 {name}，{steps} 步，batch {batch}",
            "帮我训练 {name}，用{backend_zh}，steps={steps}，batch={batch}",
        ),
        heldout=(
            "我要一个名叫 {name} 的{backend_zh}策略，训 {steps} 步、每批 {batch} 个样本",
            "开一次训练：{backend_zh}，名字 {name}，跑 {steps} 步，批量 {batch}",
        ),
    ),
    Family(  # 整族留出：这个工具在微调语料里一次也不出现
        "remove_policy",
        "remove_policy",
        lambda rng: {"policy": _pick(rng, fixture.POLICIES)},
        lambda s: {"name": s["policy"]},
        seen=(),
        heldout=(
            "把策略 {policy} 删掉",
            "从注册表里移除 {policy}",
            "{policy} 不要了，删除它和它的评测结果",
            "删除策略 {policy}",
            "清掉 {policy} 这个策略",
        ),
    ),
)

PER_TEMPLATE = {"seen": 3, "heldout": 4}  # 每条模板抽几组槽位值
_FAMILY_TOOLS = {f.name: f.tool for f in FAMILIES}


def _generated(seed: int) -> list[Case]:
    cases: list[Case] = []
    for family in FAMILIES:
        for bucket in ("seen", "heldout"):
            for t, template in enumerate(getattr(family, bucket)):
                group = f"{family.name}/{bucket}{t}"
                rng = make_rng(seed, f"aieval:{group}")
                texts: set[str] = set()
                for _ in range(50):  # 槽位少的模板抽不满，有几条算几条
                    slots = family.sample(rng)
                    text = template.format(**slots)
                    if text in texts:
                        continue
                    gold = ToolCall("gold", family.tool, family.args(slots))
                    cases.append(
                        Case(
                            f"{group}/{len(texts)}", family.name, bucket, group, (text,), (gold,), "toolcall"
                        )
                    )
                    texts.add(text)
                    if len(texts) == PER_TEMPLATE[bucket]:
                        break
    return cases


def _handwritten() -> list[Case]:
    cases = []
    for i, (family, text, tool, args) in enumerate(HANDWRITTEN):
        if _FAMILY_TOOLS.get(family, tool) != tool:
            raise ValueError(
                f"人工编写集第 {i} 条：族 {family} 的工具应是 {_FAMILY_TOOLS[family]}，写的是 {tool}"
            )
        gold = ToolCall("gold", tool, args)
        cases.append(
            Case(
                f"handwritten/{i:03d}",
                family,
                "handwritten",
                f"handwritten/{i:03d}",
                (text,),
                (gold,),
                "toolcall",
            )
        )
    return cases


def toolcall_cases(seed: int = 0) -> list[Case]:
    """toolcall 套件的全部用例。同一个种子总是得到同一组用例。"""
    return [*_generated(seed), *_handwritten()]


def suite_fingerprint(cases: Sequence[Case]) -> str:
    """用例集的指纹：用例的全部内容，与顺序无关。"""
    return sha256_text(canonical_json(sorted((asdict(c) for c in cases), key=lambda c: c["id"])))
