# CLAUDE.md

给 Claude Code 的工作说明：这个仓库是什么、按什么规矩写、现在做到哪一步。

## 这是什么

RoboArena 是一个机器人策略评测平台：训练一个神经仿真器（动作条件的世界模型），在里面批量评估机器人策略，
并检验这种评估和真实仿真器里的结果是否一致。它是「世界模型」课程 Phase 12 的结业项目，但作为**独立仓库**开发，
不导入课程仓库（`../world-models-course`）的任何代码，也不要改动那个仓库。

- 需求：[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)（需求编号 FR-* / NFR-*，里程碑，止损条件）
- 架构：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（模块、接口、表结构、协议、设计决策）
- 进度：[docs/PROGRESS.md](docs/PROGRESS.md)（**以它为准**；完成一项就更新）

动手之前先读这三份文档的相关部分。实现与文档不一致时，先指出来，改文档或改实现，不要让两者悄悄分叉。

## 常用命令

```bash
uv sync                                   # 安装依赖
uv run pytest                             # 测试（CPU，不联网）
uv run ruff check . && uv run ruff format --check .
uv run pyright                            # 静态类型检查
uv run roboarena info                     # 设备、工作区、已注册的任务
uv run roboarena --help
ROBOARENA_QUICK=1 uv run roboarena …      # 冒烟规模，等价于 --config configs/quick.toml
```

## 结构与依赖方向

```
src/roboarena/
  core/       设备、种子、哈希、日志、异常          （不依赖其他模块）
  config.py   dataclass 配置 + TOML + 覆盖          （只依赖 core）
  store/      工作区：SQLite、运行记录、注册表       （只依赖 core）
  tasks/      任务与仿真器接口；各任务的实现
  data/       回合文件、采集、校验、切分、指纹
  policies/   策略接口与实现
  worldmodel/ 世界模型接口与实现                    （M3）
  sim/        NeuralSim                             （M3）
  eval/       评测器、统计、排行榜
  serve/      FastAPI 服务                          （M5a）
  cli.py      命令行入口
```

依赖只能从上层指向下层：`cli / serve → eval → policies / sim → worldmodel → data → tasks → store / config → core`。
`tests/test_architecture.py` 会检查，违反即失败。另外几条硬规则：

- 包内一律用绝对导入（`from roboarena.core...`）。
- `policies` 与 `worldmodel` 互不导入；`sim` 不导入 `policies`。
- `gym_pusht` 只在 `tasks/pusht/` 里导入；FastAPI 只在 `serve/` 里导入。
- `tasks/` 以外的代码不出现某个任务专有的名字（`coverage`、512、5 维状态）。通用的说法是 `progress`、`TaskSpec`。
- 策略只看画面。接口边界上画面是 `uint8 [.., H, W, 3]`，动作是环境坐标；归一化是模型内部的事。

## 工作顺序

里程碑按顺序做，一个做完再做下一个：M0 骨架 → M1 数据 → M2 策略库 → M3 神经仿真器 → M4 评测与秩相关 →
M5a 服务 → M5b 前端 → M6 消融、报告、容器、持续集成。

**M4 的秩相关数字出来之前，不建前端工程。** 止损条件见需求文档第 7 节。

每个里程碑收尾时：测试、`ruff`、`pyright` 全过；实际运行一遍相关命令并看输出；更新 `docs/PROGRESS.md`。

## 写代码的规矩

- **语言**：文档、注释、docstring、程序输出、报错信息用中文；标识符用英文。
- **类型**：公开函数都有类型标注；张量形状写在注释里，如 `[B, T, H, W, 3]`。
- **数值**：张量一律 float32（MPS 不支持 float64）；设备用 `roboarena.core.device.get_device()`。
- **随机数**：不用全局随机状态。每个组件用 `derive_seed(全局种子, 组件名)` 派生自己的发生器。
- **测试**：新功能带测试。统计函数对手算例子；任务插件过 `tests/test_tasks.py` 的契约测试。测试不联网、默认跑 CPU。
- **依赖**：只加当前里程碑实际用到的包。
- **安全**：加载检查点用 `torch.load(..., weights_only=True)`；接口不接受文件路径；令牌不进日志。
- **不提前抽象**：只为已经有两个使用者的东西建接口。

## 事实与数字

- 报告、README、文档里的**实验数字只写真实跑出来的**。没跑的地方写 `> 📌 待回填：<要填什么，来自哪条命令>`。
- 目标值和实测值分开写，不要把目标写成结果。
- 负结果照实写。
- 论文引用写成「作者 年份（arXiv:编号）」；编号没核对过的标「（待核对）」。

## 环境

- Python 3.12，`uv` 管理。本机是 M1 Max 32 GB，没有 NVIDIA GPU；重训练可选 4090（`configs/cloud.toml`，尚未建立）。
- `gym-pusht` 声明依赖 `pygame` 和带界面的 `opencv-python`，已在 `[tool.uv] override-dependencies` 里排除，
  改用 `pygame-ce` 与 `opencv-python-headless`。不要用 `pip install` 绕过。
- PushT 人类演示第一次用到时下载到 `workspace/cache/pusht/raw/`（约 8 MB）。访问不了 huggingface.co 时设
  `HF_ENDPOINT=https://hf-mirror.com`。依赖演示数据的测试在没有缓存时自动跳过。
- `gym-pusht` 的一个坑：`reset_to_state` 设定朝向时块的位置会偏移，传入的块坐标不等于重置后读到的位姿。
- `workspace/` 是运行产物，不入库。

## 不要做的事

- 不要自动 `git init`、`git commit`、`git push`，除非用户明确要求。
- 不要写没跑过的实验数字。
- 不要改动 `../world-models-course`。
- 不要并行派发子代理同时做多个里程碑。
