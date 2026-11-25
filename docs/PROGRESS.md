# 开发进度

> 最后更新：2026-11-25 ｜ 当前里程碑：**M1 数据**（未开始）
>
> 状态：✅ 完成并验证 ｜ 🚧 进行中 ｜ ⏳ 未开始。需求编号见 [REQUIREMENTS.md](./REQUIREMENTS.md)。

## 总览

| 里程碑 | 内容 | 状态 | 完成的标志 |
| --- | --- | --- | --- |
| D0 | 需求文档与架构文档 | ✅ | v0.2 评审通过 |
| M0 | 骨架、配置、运行记录、注册表、命令行、任务插件接口 | ✅ | 玩具任务走通评测流程；测试通过 |
| M1 | 数据管线 | ⏳ | 数据集与评测集生成，校验通过，有指纹 |
| M2 | 策略库 | ⏳ | 全部策略在真实仿真器里评完，水平拉开 |
| M3 | 神经仿真器 | ⏳ | 一步预测优于复制上一帧；判定头误差有报告 |
| M4 | 评测与一致性分析 | ⏳ | 得到秩相关及其区间 |
| M5a | 服务、鉴权、可观测性 | ⏳ | 接口测试通过；脚本客户端完成一次遥操作 |
| M5b | React 前端 | ⏳ | 浏览器里能看排行榜、能推一回合 |
| M6 | 消融、报告、容器、持续集成 | ⏳ | 报告写完；持续集成全绿 |

硬约束：M4 的秩相关数字出来之前，不建前端工程。

## M0 明细（✅ 2026-11-25）

| 项 | 位置 | 对应需求 |
| --- | --- | --- |
| 包骨架、依赖声明、`ruff` 与 `pyright` 配置 | `pyproject.toml` | NFR-6、NFR-8 |
| 设备、派生种子、内容哈希、JSON 日志（带追踪编号） | `src/roboarena/core/` | NFR-1、FR-O1 的基础 |
| 配置：默认值 → TOML → `--set 键=值`；未知键报错 | `src/roboarena/config.py`、`configs/` | FR-M1、FR-M5（冒烟、本机两档） |
| 工作区与全部表结构（含 `jobs`、`tokens`，暂未使用） | `store/db.py`、`store/workspace.py` | 架构 6.2 |
| 运行记录：配置快照、代码版本、锁文件摘要、指标、日志、状态 | `store/runs.py` | FR-M2 |
| 注册表：按内容寻址存检查点，加载前核对哈希 | `store/registry.py` | FR-M4 |
| 评测集与评测记录：指纹、唯一键、逐回合结果 | `store/suites.py`、`store/evals.py` | FR-D6（存取部分）、FR-E6 |
| 任务与仿真器接口；玩具任务 | `tasks/api.py`、`tasks/toy.py` | FR-T1、FR-T2、FR-T4、FR-T5 |
| 策略接口、后端注册、随机策略 | `policies/` | FR-P1、FR-P5（接口部分） |
| 评测循环（动作分块、批量、提前结束）；Wilson 区间；排行榜 | `eval/` | FR-E1、FR-E2 |
| 命令行：`info`、`task list`、`data suite`、`policy add/list`、`eval run/leaderboard`、`runs list` | `cli.py` | — |
| 架构测试：依赖方向、第三方库的位置、任务专有名字 | `tests/test_architecture.py` | NFR-6 |

验证（本机实际运行）：

- `uv run pytest`：64 项通过。
- `uv run ruff check .`、`uv run ruff format --check .`、`uv run pyright`：均无报错。
- 冒烟配置下手动走了一遍 `info → data suite → policy add → eval run → eval leaderboard → runs list`，
  运行目录里生成了 `config.toml`、`meta.json`、`metrics.jsonl`、`log.txt`。

M0 的已知缺口与说明：

- 尚未 `git init`，所以运行记录里的代码版本目前是空的；建仓库后自动生效。
- 断点续训（FR-M3）要等有训练循环时再做，归入 M2。
- `configs/cloud.toml` 尚未建立，等有模型参数时再加。
- `ruff` 的行宽检查（E501）关闭，行宽交给格式化器；中文长字符串不强行折行。
- `train`、`report`、`pipeline`、`bench`、`serve`、`token` 是占位命令，运行时提示计划在哪个里程碑实现。

## 下一步：M1 数据

1. 加入 `gym-pusht` 等依赖（按 CLAUDE.md 的写法排除 `pygame` 与带界面的 OpenCV），实现 `tasks/pusht/` 并通过任务契约测试。
2. 移植人类演示的下载与解码；统一的回合文件格式。
3. 仿真器内采集带标签的回合（加噪回放、随机游走）。
4. 校验、按回合切分、指纹、清单。

## 待回填

- 策略排序的秩相关与区间、可信视野（M4）
- 神经仿真器单步延迟、遥操作帧率（M5a）
- 各阶段训练耗时（M2、M3）
