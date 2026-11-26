# RoboArena

机器人策略评测平台：训练一个神经仿真器（动作条件的世界模型），在里面评估机器人策略，
并检验这种评估与真实仿真器里的结果是否一致。平台之上计划加一层大模型应用（用自然语言操作平台的 Agent、模型网关、
AI 评测、领域小模型的微调与部署），设计见架构文档第 16 节，尚未实现。

当前进度见 [docs/PROGRESS.md](docs/PROGRESS.md)：数据管线与策略库已完成，神经仿真器与应用层还没有。
需求、架构与实验计划见 [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)、[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)、[docs/PLAN.md](docs/PLAN.md)。

## 快速开始

```bash
uv sync
uv run pytest

# 用玩具任务走一遍评测流程
export ROBOARENA_QUICK=1
uv run roboarena info
uv run roboarena data build      # 演示 + 仿真器内采集，校验、切分、算指纹
uv run roboarena data suite
uv run roboarena policy add --name random --backend random
uv run roboarena eval run --policy random
uv run roboarena eval leaderboard
```

## 目前的结果

17 个策略在真实仿真器里的成功率（PushT，50 个固定的初始状态，本机 M1 Max 实测；完整的表与区间见进度文档）：

| 策略 | 成功 | 成功率 | 95% Wilson 区间 |
| --- | --- | --- | --- |
| `bc`（行为克隆，164 条演示，20000 步） | 23/50 | 0.460 | [0.330, 0.596] |
| `bc-step10000`（训练到一半的快照） | 16/50 | 0.320 | [0.208, 0.458] |
| `dp`（Diffusion Policy） | 10/50 | 0.200 | [0.112, 0.330] |
| `bc-demos50`（只用 50 条演示） | 9/50 | 0.180 | [0.098, 0.308] |
| `random`（随机动作） | 0/50 | 0.000 | [0.000, 0.071] |

> 📌 待回填：策略排序的秩相关、可信视野、延迟，来自 M4 之后的 `roboarena report`。
>
> 📌 待回填：Agent 的任务完成率与幻觉率、微调前后的对比，来自 A1–A3 的 `roboarena aieval report`。

PushT 上的完整流程（本机 M1 Max）：

```bash
C="--config configs/local.toml"
uv run roboarena data build $C       # 数据集：演示 + 加噪重放 + 随机游走，约 2.5 分钟（首次会下载约 8 MB 演示）
uv run roboarena data suite $C       # 评测集：50 个固定的初始状态
uv run roboarena train zoo $C        # 策略库：行为克隆及其变体、Diffusion Policy、随机策略
uv run roboarena eval run --all $C   # 在真实仿真器里评测全部策略
uv run roboarena eval leaderboard $C
uv run roboarena data rollouts $C    # 把策略自己的回合记录下来，供世界模型训练
uv run roboarena data build $C --set data.name=full --set 'data.rollouts=["zoo"]'   # 并入策略回合的数据集
```

训练中断后可以续训：`uv run roboarena train policy --resume <运行编号>`；`train zoo` 重新运行会跳过已有的策略。
