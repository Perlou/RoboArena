"""架构规则（docs/ARCHITECTURE.md 第 3 节与 16.1 节）：依赖只能从上层指向下层。扫描导入语句检查。"""

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "roboarena"

# 层级：只能导入层级严格更低的模块（或自己所在的模块）
RANK = {
    "core": 0,
    "config": 1,
    "store": 1,
    "tasks": 2,
    "data": 3,
    "worldmodel": 4,
    "policies": 5,
    "sim": 5,
    "eval": 6,
    "llm": 7,
    "agent": 8,
    "aieval": 9,
    "tune": 10,
    "serve": 11,
    "cli": 11,
}
# 层级允许、但另有规定不许的导入
FORBIDDEN = {("policies", "worldmodel")}
# 只许导入列出的单元：模型网关不该知道平台的存在
ONLY = {"llm": {"core", "config"}}
# 第三方库只许出现在指定位置
THIRD_PARTY = {
    "fastapi": "serve/",
    "gym_pusht": "tasks/pusht/",
    "gymnasium": "tasks/pusht/",
    "httpx": "llm/",
    "transformers": "tune/",
    "peft": "tune/",
}
# 某个任务专有的名字，不许出现在与任务无关的模块里
TASK_WORDS = ("coverage", "pusht", "block_pose")
TASK_AGNOSTIC = ("data", "policies", "worldmodel", "sim", "eval", "llm", "agent", "aieval", "tune", "serve")


def _unit(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from ((0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.level, node.module or ""


def _files():
    return [p for p in sorted(SRC.rglob("*.py")) if _unit(p) != "__init__"]


def test_every_unit_has_a_rank():
    assert {_unit(p) for p in _files()} <= set(RANK)


def _import_problems(where: str, imports: Iterable[tuple[int, str]]) -> list[str]:
    """一个文件里违反规则的导入。where 是相对 src/roboarena 的路径（正斜杠）。"""
    unit = where.split("/")[0] if "/" in where else where.removesuffix(".py")
    problems = []
    for level, module in imports:
        if level:
            problems.append(f"{where}：包内请用绝对导入")
            continue
        parts = module.split(".")
        allowed = THIRD_PARTY.get(parts[0])
        if allowed is not None and not where.startswith(allowed):
            problems.append(f"{where}：{module} 只能在 {allowed} 里导入")
        if parts[0] != "roboarena" or len(parts) < 2 or parts[1] == unit:
            continue
        target = parts[1]
        blocked = (unit, target) in FORBIDDEN or target not in ONLY.get(unit, RANK)
        if RANK[target] >= RANK[unit] or blocked:
            problems.append(f"{where}：{unit} 不能导入 {target}")
    return problems


def test_imports_follow_the_rules():
    problems = []
    for path in _files():
        problems += _import_problems(path.relative_to(SRC).as_posix(), _imports(path))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "where, module, message",
    [
        ("policies/bc.py", "roboarena.worldmodel.api", "policies 不能导入 worldmodel"),
        ("eval/service.py", "roboarena.agent.tools", "eval 不能导入 agent"),
        ("agent/tools.py", "roboarena.aieval.cases", "agent 不能导入 aieval"),
        ("llm/gateway.py", "roboarena.store.runs", "llm 不能导入 store"),
        ("agent/tools.py", "httpx", "httpx 只能在 llm/ 里导入"),
        ("cli.py", "httpx", "httpx 只能在 llm/ 里导入"),
    ],
)
def test_the_checker_catches_violations(where, module, message):
    assert any(message in p for p in _import_problems(where, [(0, module)]))
    assert _import_problems(where, [(1, "api")]) == [f"{where}：包内请用绝对导入"]


@pytest.mark.parametrize(
    "where, module",
    [
        ("llm/gateway.py", "roboarena.config"),
        ("llm/gateway.py", "roboarena.core.hashing"),
        ("llm/openai_compat.py", "httpx"),
        ("agent/tools.py", "roboarena.eval.service"),
        ("agent/loop.py", "roboarena.llm.api"),
        ("aieval/run.py", "roboarena.agent.tools"),
        ("tune/corpus.py", "roboarena.aieval.cases"),
        ("cli.py", "roboarena.aieval.run"),
        ("serve/app.py", "roboarena.agent.loop"),
    ],
)
def test_the_checker_accepts_allowed_imports(where, module):
    assert _import_problems(where, [(0, module)]) == []


def test_task_specific_names_stay_in_tasks():
    problems = []
    for path in _files():
        if _unit(path) in TASK_AGNOSTIC:
            text = path.read_text(encoding="utf-8").lower()
            problems += [
                f"{path.relative_to(SRC)}：出现了任务专有的名字 {w}" for w in TASK_WORDS if w in text
            ]
    assert not problems, "\n".join(problems)
