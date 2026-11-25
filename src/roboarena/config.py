"""配置：dataclass 默认值 → TOML 文件 → 命令行的 键=值 覆盖，后者覆盖前者。

未知的键直接报错，避免拼错的配置项被悄悄忽略。
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from roboarena.core.device import is_quick
from roboarena.core.errors import ConfigError
from roboarena.core.hashing import canonical_json, sha256_text


@dataclass
class WorkspaceConfig:
    root: str = "workspace"


@dataclass
class TaskConfig:
    name: str = "pusht"


@dataclass
class EvalConfig:
    suite: str = "main"  # 评测集的名称
    n_inits: int = 50  # 评测集里初始状态的个数（只在生成评测集时用）
    max_steps: int = 0  # 每回合步数上限；0 表示用任务自己的上限
    n_exec: int = 1  # 每次调用策略后，执行动作分块里的前几个动作
    batch: int = 16  # 并行评测的回合数
    seed: int = 0  # 评测时策略的随机种子


@dataclass
class Config:
    seed: int = 0
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_toml(self) -> str:
        return dump_toml(self.to_dict())

    def digest(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


def _coerce(value: Any, expected: type, key: str) -> Any:
    if expected is float and type(value) is int:
        return float(value)
    if type(value) is not expected:
        raise ConfigError(f"配置项 {key} 应为 {expected.__name__}，实际是 {type(value).__name__}：{value!r}")
    return value


def _apply(obj: Any, data: Mapping[str, Any], prefix: str = "") -> None:
    hints = get_type_hints(type(obj))
    for key, value in data.items():
        if key not in hints:
            raise ConfigError(f"未知的配置项：{prefix}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ConfigError(f"配置项 {prefix}{key} 应为一个表")
            _apply(current, value, f"{prefix}{key}.")
        else:
            setattr(obj, key, _coerce(value, hints[key], prefix + key))


def _parse_override(item: str) -> dict[str, Any]:
    """把 "eval.max_steps=50" 变成 {"eval": {"max_steps": 50}}。值按 TOML 解析，解析不了当作字符串。"""
    if "=" not in item:
        raise ConfigError(f"覆盖项应写成 键=值：{item!r}")
    key, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    nested: dict[str, Any] = {}
    cursor = nested
    parts = key.strip().split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return nested


def config_dir() -> Path:
    """预设配置所在的目录：环境变量 → 当前目录下的 configs → 仓库里的 configs。"""
    if env := os.environ.get("ROBOARENA_CONFIG_DIR"):
        return Path(env)
    local = Path.cwd() / "configs"
    return local if local.is_dir() else Path(__file__).resolve().parents[2] / "configs"


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> Config:
    """读取配置。没有指定文件且处于冒烟模式（ROBOARENA_QUICK=1）时，用 configs/quick.toml。"""
    if path is None and is_quick():
        path = config_dir() / "quick.toml"
    cfg = Config()
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"找不到配置文件：{path}")
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"配置文件 {path} 不是合法的 TOML：{e}") from e
        _apply(cfg, data)
    for item in overrides:
        _apply(cfg, _parse_override(item))
    return cfg


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ConfigError(f"无法写成 TOML 的值：{value!r}")


def dump_toml(data: Mapping[str, Any], prefix: str = "") -> str:
    """把嵌套字典写成 TOML（标准库只有读，没有写）。只支持配置里用到的类型。"""
    lines = [f"{k} = {_toml_value(v)}" for k, v in data.items() if not isinstance(v, Mapping)]
    blocks = ["\n".join(lines)] if lines else []
    for key, value in data.items():
        if isinstance(value, Mapping):
            name = f"{prefix}{key}"
            body = dump_toml(value, f"{name}.")
            blocks.append(f"[{name}]\n{body}".rstrip())
    return "\n\n".join(blocks) + "\n"
