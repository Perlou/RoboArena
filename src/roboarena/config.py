"""配置：dataclass 默认值 → TOML 文件 → 命令行的 键=值 覆盖，后者覆盖前者。

未知的键直接报错，避免拼错的配置项被悄悄忽略。
"""

from __future__ import annotations

import copy
import inspect
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

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
class DataConfig:
    name: str = "main"  # 数据集的名称
    n_demos: int = -1  # 用多少条任务自带的演示；-1 表示全部，0 表示不用
    n_replay: int = 300  # 演示动作加噪、从随机初始状态重放的回合数
    replay_noise: float = 0.03  # 噪声标准差占动作范围的比例；每个回合在 0 到 2 倍之间取一个强度
    n_random_walk: int = 200  # 随机游走的回合数
    random_walk_steps: int = 150  # 每个随机游走回合的步数
    val_frac: float = 0.1
    test_frac: float = 0.1
    batch: int = 16  # 采集时并行的回合数
    rollouts: list[str] = field(default_factory=list)  # 并入哪些策略回合集（roboarena data rollouts 生成）


@dataclass
class RolloutConfig:
    """策略回合集：让已登记的策略在真实仿真器里跑，记录下来当训练数据（主要给世界模型用）。"""

    name: str = "zoo"
    policies: list[str] = field(default_factory=list)  # 空表示当前任务的全部策略
    per_policy: int = 16  # 每个策略跑多少个回合
    extra_steps: int = 10  # 成功之后再多记录几步


@dataclass
class PolicyConfig:
    """可训练策略的结构与训练参数。行为克隆用 [policy] 段，Diffusion Policy 用 [diffusion] 段。"""

    obs_frames: int = 2  # 输入的历史帧数
    chunk: int = 8  # 一次预测的动作个数
    width: int = 32  # 卷积通道数的基数
    hidden: int = 512  # 全连接层宽度
    n_demos: int = -1  # 用训练集里的多少条演示；-1 表示全部
    steps: int = 20000
    batch: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup: int = 500  # 学习率线性升温的步数，之后余弦衰减到 0
    aug_shift: int = 4  # 训练时画面随机平移的最大像素数；0 表示不做
    log_every: int = 100
    eval_every: int = 1000  # 每隔多少步算一次验证损失并保存续训检查点
    val_batches: int = 20
    snapshots: list[int] = field(default_factory=list)  # 在这些步数把当时的权重另存并登记成独立的策略
    ema: float = 0.0  # 权重的指数滑动平均；大于 0 时登记的是平均后的权重。扩散模型一般要开
    diffusion_steps: int = 100  # 只对 Diffusion Policy 有效：训练时的加噪步数
    sample_steps: int = 10  # 只对 Diffusion Policy 有效：采样时 DDIM 走几步


@dataclass
class ZooConfig:
    """策略库：一组水平不同的策略。roboarena train zoo 按它训练并登记。"""

    diffusion: bool = (
        False  # 是否另外训练一个 Diffusion Policy（名称 dp，按 [diffusion] 段的 snapshots 留快照）
    )
    base: str = "bc"  # 用全部演示训练的那个策略的名称；其余策略的名称由它派生
    n_demos: list[int] = field(default_factory=list)  # 另外用更少的演示各训练一个
    exec_noise: list[float] = field(
        default_factory=list
    )  # 给 base 的动作加噪声，各登记一个；标准差占动作范围的比例


@dataclass
class EvalConfig:
    suite: str = "main"  # 评测集的名称
    n_inits: int = 50  # 评测集里初始状态的个数（只在生成评测集时用）
    max_steps: int = 0  # 每回合步数上限；0 表示用任务自己的上限
    n_exec: int = 1  # 每次调用策略后，执行动作分块里的前几个动作
    batch: int = 16  # 并行评测的回合数
    seed: int = 0  # 评测时策略的随机种子


@dataclass
class LLMModelConfig:
    """一个语言模型的接入方式。密钥不写在这里：api_key_env 是存放密钥的环境变量的名字。"""

    backend: str = "openai"  # 目前只有 openai：OpenAI 兼容的 HTTP 接口
    base_url: str = "http://127.0.0.1:11434/v1"  # 本机 Ollama 的 OpenAI 兼容接口
    model: str = ""  # 推理服务里的模型标识
    api_key_env: str = ""  # 密钥所在的环境变量名；本机服务不需要
    price_in: float = 0.0  # 每百万输入 token 的价格
    price_out: float = 0.0  # 每百万输出 token 的价格


def _default_llm_models() -> dict[str, LLMModelConfig]:
    return {
        "qwen7b": LLMModelConfig(model="qwen2.5:7b"),
        "qwen08b": LLMModelConfig(model="qwen3.5:0.8b"),
    }


@dataclass
class LLMConfig:
    """模型网关。models 是「名称 → 接入方式」，其余各处只用名称指代模型。"""

    default: str = "qwen7b"  # chat 与 aieval 不指定模型时用它
    fallback: str = ""  # 重试后仍失败时改用它；空表示不降级
    timeout_s: float = 120.0  # 单次请求的超时
    retries: int = 2  # 超时或服务端报错时再试几次
    cache: bool = True  # 相同的请求直接取上次的回答；测延迟时关掉
    models: dict[str, LLMModelConfig] = field(default_factory=_default_llm_models)


@dataclass
class Config:
    seed: int = 0
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollouts: RolloutConfig = field(default_factory=RolloutConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    diffusion: PolicyConfig = field(default_factory=lambda: PolicyConfig(ema=0.999))
    zoo: ZooConfig = field(default_factory=ZooConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_toml(self) -> str:
        return dump_toml(self.to_dict())

    def digest(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


LLM_BACKENDS = ("openai",)
_NAME = re.compile(r"[A-Za-z0-9_-]+")


def _coerce(value: Any, expected: Any, key: str) -> Any:
    if get_origin(expected) is list:
        if not isinstance(value, list):
            raise ConfigError(f"配置项 {key} 应为列表，实际是 {type(value).__name__}：{value!r}")
        (item_type,) = get_args(expected)
        return [_coerce(v, item_type, f"{key}[{i}]") for i, v in enumerate(value)]
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
        elif get_origin(hints[key]) is dict:
            _apply_named(current, value, get_args(hints[key])[1], prefix + key)
        else:
            setattr(obj, key, _coerce(value, hints[key], prefix + key))


def _apply_named(table: dict[str, Any], data: Any, item_type: type, key: str) -> None:
    """「名称 → 表」的配置项：已有的名称只改提到的字段，新名称先按默认值建一项。"""
    if not isinstance(data, Mapping):
        raise ConfigError(f"配置项 {key} 应为一个表")
    for name, value in data.items():
        if not _NAME.fullmatch(name):
            raise ConfigError(f"配置项 {key} 里的名称 {name} 只能由字母、数字、下划线、连字符组成")
        if not isinstance(value, Mapping):
            raise ConfigError(f"配置项 {key}.{name} 应为一个表")
        entry = table.get(name) or item_type()
        _apply(entry, value, f"{key}.{name}.")
        table[name] = entry


def _validate(cfg: Config) -> None:
    """单个配置项的类型没问题之后，再查各项之间是否自洽。"""
    llm = cfg.llm
    for key, name in (("default", llm.default), ("fallback", llm.fallback)):
        if name and name not in llm.models:
            known = "、".join(llm.models) or "（空）"
            raise ConfigError(
                f"配置项 llm.{key} 指的模型 {name} 没有在 llm.models 里定义；已定义的有：{known}"
            )
    if not llm.default:
        raise ConfigError("配置项 llm.default 不能为空")
    if llm.retries < 0:
        raise ConfigError(f"配置项 llm.retries 不能是负数：{llm.retries}")
    if llm.timeout_s <= 0:
        raise ConfigError(f"配置项 llm.timeout_s 应大于 0：{llm.timeout_s}")
    for name, model in llm.models.items():
        if model.backend not in LLM_BACKENDS:
            raise ConfigError(
                f"配置项 llm.models.{name}.backend 应为 {'、'.join(LLM_BACKENDS)} 之一，实际是 {model.backend!r}"
            )


def split_override(item: str) -> tuple[str, Any]:
    """把 "eval.max_steps=50" 拆成（键，值）。值按 TOML 解析，解析不了当作字符串。"""
    if "=" not in item:
        raise ConfigError(f"覆盖项应写成 键=值：{item!r}")
    key, raw = item.split("=", 1)
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return key.strip(), value


def _parse_override(item: str) -> dict[str, Any]:
    """把 "eval.max_steps=50" 变成 {"eval": {"max_steps": 50}}。值按 TOML 解析，解析不了当作字符串。"""
    key, value = split_override(item)
    nested: dict[str, Any] = {}
    cursor = nested
    parts = key.split(".")
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
    _validate(cfg)
    return cfg


def with_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """在 cfg 的副本上应用覆盖项并校验，原来的 cfg 不变。未知的键、类型不对都会报错。"""
    changed = copy.deepcopy(cfg)
    for item in overrides:
        _apply(changed, _parse_override(item))
    _validate(changed)
    return changed


def _field_docs(cls: type) -> dict[str, str]:
    """字段的说明：取源码里字段定义后面的行内注释。"""
    names = set(get_type_hints(cls))
    docs: dict[str, str] = {}
    current: str | None = None
    for line in inspect.getsource(cls).splitlines():
        if (m := re.match(r"\s+(\w+)\s*:", line)) and m.group(1) in names:
            current = m.group(1)
        if current is not None and current not in docs and "#" in line:
            docs[current] = line.split("#", 1)[1].strip()
    return docs


def describe_config(section: str = "") -> list[dict[str, Any]]:
    """配置项的清单：键、类型、默认值、说明。section 给出时只列这一段（如 "policy"）。"""
    found: list[dict[str, Any]] = []

    def walk(obj: Any, prefix: str) -> None:
        docs = _field_docs(type(obj))
        for name, hint in get_type_hints(type(obj)).items():
            value = getattr(obj, name)
            if is_dataclass(value):
                walk(value, f"{prefix}{name}.")
            elif not isinstance(value, dict):  # 「名称 → 表」的项不在这里展开
                kind = hint.__name__ if isinstance(hint, type) else str(hint)
                found.append(
                    {"key": prefix + name, "type": kind, "default": value, "doc": docs.get(name, "")}
                )

    walk(Config(), "")
    if not section:
        return found
    chosen = [f for f in found if f["key"] == section or f["key"].startswith(section + ".")]
    if not chosen:
        sections = "、".join(n for n, v in vars(Config()).items() if is_dataclass(v))
        raise ConfigError(f"没有名为 {section} 的配置段；有的是：{sections}")
    return chosen


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
