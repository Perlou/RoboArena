"""架构规则（docs/ARCHITECTURE.md 第 3 节）：依赖只能从上层指向下层。扫描导入语句检查。"""

import ast
from pathlib import Path

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
    "serve": 7,
    "cli": 7,
}
# 层级允许、但另有规定不许的导入
FORBIDDEN = {("policies", "worldmodel")}
# 第三方库只许出现在指定位置
THIRD_PARTY = {"fastapi": "serve/", "gym_pusht": "tasks/pusht/", "gymnasium": "tasks/pusht/"}
# 某个任务专有的名字，不许出现在与任务无关的模块里
TASK_WORDS = ("coverage", "pusht", "block_pose")
TASK_AGNOSTIC = ("data", "policies", "worldmodel", "sim", "eval", "serve")


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


def test_dependency_direction():
    problems = []
    for path in _files():
        unit = _unit(path)
        for level, module in _imports(path):
            where = path.relative_to(SRC)
            if level:
                problems.append(f"{where}：包内请用绝对导入")
                continue
            parts = module.split(".")
            if parts[0] != "roboarena" or len(parts) < 2 or parts[1] == unit:
                continue
            target = parts[1]
            if RANK[target] >= RANK[unit] or (unit, target) in FORBIDDEN:
                problems.append(f"{where}：{unit} 不能导入 {target}")
    assert not problems, "\n".join(problems)


def test_third_party_imports_stay_in_their_place():
    problems = []
    for path in _files():
        where = path.relative_to(SRC).as_posix()
        for _, module in _imports(path):
            allowed = THIRD_PARTY.get(module.split(".")[0])
            if allowed is not None and not where.startswith(allowed):
                problems.append(f"{where}：{module} 只能在 {allowed} 里导入")
    assert not problems, "\n".join(problems)


def test_task_specific_names_stay_in_tasks():
    problems = []
    for path in _files():
        if _unit(path) in TASK_AGNOSTIC:
            text = path.read_text(encoding="utf-8").lower()
            problems += [
                f"{path.relative_to(SRC)}：出现了任务专有的名字 {w}" for w in TASK_WORDS if w in text
            ]
    assert not problems, "\n".join(problems)
