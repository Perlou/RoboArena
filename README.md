# RoboArena

机器人策略评测平台：训练一个神经仿真器（动作条件的世界模型），在里面评估机器人策略，
并检验这种评估与真实仿真器里的结果是否一致。

项目处于早期。当前进度见 [docs/PROGRESS.md](docs/PROGRESS.md)，
需求与架构见 [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)、[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

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

> 📌 待回填：研究结果（策略排序的秩相关、可信视野、延迟），来自 M4 之后的 `roboarena report`。

PushT 的数据集（本机约两分半钟，首次运行会下载约 8 MB 的人类演示）：

```bash
uv run roboarena data build --config configs/local.toml
uv run roboarena data suite --config configs/local.toml
```
