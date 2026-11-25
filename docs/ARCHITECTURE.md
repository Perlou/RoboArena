# RoboArena 架构文档

> 版本：v0.2（已纳入评审结论，待确认） ｜ 日期：2026-11-25 ｜ 配套文档：[REQUIREMENTS.md](./REQUIREMENTS.md)
>
> 本文档描述系统怎么搭：模块、接口、数据格式、存储、服务协议、部署。
> 模型的具体宽度、层数等超参数不在这里定，它们在最小版本跑出结果后写进配置文件。

## 1. 系统总览

```
┌────────────────────────────── 离线：训练与评测（命令行） ───────────────────────────────┐
│                                                                                      │
│  人类演示 ─┐                                                                          │
│           ├─▶ 数据管线 ─▶ 数据集（回合文件 + 指纹）─┬─▶ 训练策略 ─────▶ 策略检查点       │
│  仿真采集 ─┘      │                                │                                  │
│                  └─▶ 评测集（一组初始状态）          └─▶ 训练世界模型 ─▶ 世界模型检查点    │
│                           │                                                          │
│                           ▼                                                          │
│                  ┌──────────────────┐   同一个策略、同一组初始状态                     │
│                  │      评测器       │──▶ RealSim  （gym-pusht）   ──▶ 真实成功率       │
│                  │ （与仿真器无关）   │──▶ NeuralSim（世界模型）     ──▶ 模型内成功率     │
│                  └──────────────────┘                    │                           │
│                                                          ▼                           │
│                                        一致性分析：秩相关、MMRV、开环误差、可信视野      │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │ 全部写入
                                           ▼
                   ┌────────────────────────────────────────────────┐
                   │  工作区 workspace/                              │
                   │  roboarena.db（SQLite）＋ runs/ ＋ artifacts/    │
                   └───────────────────────┬────────────────────────┘
                                           │ 读取
┌──────────────────────────────────────────┴───────────── 在线：服务 ───────────────────┐
│  FastAPI（令牌鉴权 ｜ 结构化日志 ｜ 指标端点）                                          │
│   ├─ REST：策略 / 仿真器 / 排行榜 / 一致性 / 运行记录 / 评测任务                         │
│   ├─ 任务执行器：后台线程串行执行评测任务                                               │
│   └─ WebSocket：遥操作会话（动作进，画面出）                                            │
│                                                                                      │
│  浏览器（React 单页应用）：排行榜 ｜ 一致性 ｜ 遥操作                                    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

离线部分和在线部分只通过工作区耦合：服务不训练模型，只读结果、跑评测、驱动遥操作。
没有服务，命令行也能得到全部结果；这保证研究结论不依赖界面。

图里的「gym-pusht」是当前唯一的任务。任务以插件形式接入（第 4.1 节），评测器、服务、前端都不认识具体任务。

## 2. 仓库结构

```
RoboArena/
├── README.md                  项目报告（结果只填实测）
├── pyproject.toml / uv.lock / .python-version      Python 3.12，uv 管理
├── Dockerfile
├── .github/workflows/          持续集成
├── docs/
│   ├── REQUIREMENTS.md
│   ├── ARCHITECTURE.md
│   └── PLAN.md                一页纸实验计划：预注册的消融、可信视野的阈值（M4 之前写定）
├── configs/
│   ├── quick.toml             冒烟：数分钟跑完全流程
│   ├── local.toml             本机 M1 Max 的标准规模
│   └── cloud.toml             4090 的放大规模
├── src/roboarena/
│   ├── cli.py                 命令行入口
│   ├── config.py              配置
│   ├── core/                  设备、种子、哈希、日志
│   ├── tasks/                 任务与仿真器接口；pusht/（真实任务及其真实仿真器）；toy/（测试用的玩具任务）
│   ├── store/                 工作区：数据库、运行记录、注册表
│   ├── data/                  下载、采集、回合文件、校验、切分、指纹、采样
│   ├── sim/                   NeuralSim：把世界模型包装成仿真器
│   ├── worldmodel/            世界模型接口、latent 后端、训练
│   ├── policies/              策略接口、行为克隆、Diffusion Policy、随机策略、训练
│   ├── eval/                  评测器、统计、一致性分析、诊断、报告
│   └── serve/                 FastAPI 应用、鉴权、指标、任务执行器、遥操作会话
├── web/                       前端工程（React + Vite + TypeScript）
│   ├── src/api/               由 OpenAPI 生成的类型 + 请求封装
│   ├── src/pages/             排行榜、一致性、遥操作
│   ├── src/components/        图表、画布、通用组件
│   └── dist/                  构建产物，由后端托管，不入库
├── tests/
└── workspace/                 运行产物，不入库
```

采用 `src/` 布局的可安装包（`uv` 以可编辑方式安装），命令行通过 `roboarena` 入口调用。
这和课程仓库「不是可安装的包、靠 `sys.path` 导入」的做法不同：本项目是独立软件，测试和服务都需要正常的包导入。

## 3. 模块与依赖方向

```
            cli            serve
              │              │
              └──────┬───────┘
                     ▼
                   eval
            ┌────────┼─────────┐
            ▼        ▼         ▼
        policies    sim ──▶ worldmodel
            │        │         │
            └────────┼─────────┘
                     ▼
                   data
                     │
                     ▼
                   tasks
                     │
                     ▼
               store ──▶ core ◀── config
```

规则：箭头只能向下。具体地说：

- `core`、`config` 不依赖任何其他模块。
- `policies` 与 `worldmodel` 互不依赖。策略不知道世界模型的存在。
- `tasks` 定义任务接口并存放各任务的实现。`tasks/pusht/` 是唯一导入 `gym_pusht` 的地方。
- `Simulator`、`StepResult`、`InitState` 这几个类型定义在 `tasks` 里（任务要返回真实仿真器，接口放在 `sim` 会形成循环依赖）。
  `sim` 只放与任务无关的 `NeuralSim`，并把这些类型原样导出；真实仿真器由任务插件构造。
- `policies`、`worldmodel`、`eval`、`serve` 里不出现 PushT 专有的名字（`coverage`、512、5 维状态）。
- `eval` 只通过 `Simulator` 和 `Policy` 两个接口工作，不导入具体的模型类。
- `serve` 不包含任何评测或统计逻辑，只做调度和序列化。
- FastAPI 只在 `serve` 里导入。

有一条测试专门检查这些导入规则（扫描各模块的 `import` 语句）。

## 4. 核心接口

张量形状约定：`B` 批大小（并行回合数），`K` 上下文帧数，`T` 时间长度。
`A` 是动作维度，`H × W` 是画面尺寸，两者由任务给出（PushT：`A = 2`，96×96）。分块长度在形状里写作 `chunk`。
画面在接口边界上一律是 `uint8`、`[..., H, W, 3]`、取值 0–255；动作在接口边界上一律是环境坐标（PushT：0–512）。
归一化是各模型内部的事，不出现在接口上。

### 4.1 Task

```python
@dataclass(frozen=True)
class TaskSpec:
    name: str  # "pusht"
    frame_shape: tuple[int, int, int]  # (96, 96, 3)
    action_low: np.ndarray  # [A]
    action_high: np.ndarray  # [A]
    control_hz: float  # 10
    max_steps: int  # 300
    label_fields: dict[str, int]  # 判定头要回归的量及其维度，PushT：{"block_pose": 3}


class Task(Protocol):
    spec: TaskSpec

    def sample_inits(self, n: int, seed: int) -> list[InitState]: ...
    def make_real_sim(self) -> Simulator: ...  # 回合数在 reset 时由 inits 的个数决定
    def encode_init(self, init: InitState) -> dict: ...  # 序列化成 JSON，存进评测集
    def decode_init(self, data: dict) -> InitState: ...
    def demos(self, cache_dir: Path, limit: int | None = None) -> list[Episode]: ...  # 任务自带的演示
```

`InitState` 对评测器是不透明的：评测器只把它从评测集里取出来交给仿真器。
PushT 的 `InitState` 是（推杆 x、y，块 x、y，块朝向，种子）；玩具任务的可以只是一个整数。

任务通过名称注册在 `tasks` 的工厂表里。`toy` 任务是一个几行代码的二维点到达问题，自带渲染，
不依赖任何物理引擎，用来在测试里走通全流程，同时检验这套抽象没有夹带 PushT 的假设。

### 4.2 Simulator

```python
@dataclass
class StepResult:
    frames: np.ndarray  # [B, H, W, 3] uint8
    progress: np.ndarray  # [B] float32，0–1 的任务进度；PushT 里就是 coverage
    success: np.ndarray  # [B] bool
    labels: dict[str, np.ndarray]  # 任务声明的标签，如 {"block_pose": [B, 3]}；神经仿真器里是读出值


class Simulator(Protocol):
    name: str

    def reset(self, inits: Sequence[InitState]) -> StepResult: ...  # 初始时刻的画面、进度与标签
    def step(self, actions: np.ndarray) -> StepResult: ...  # actions [B, A]
    def close(self) -> None: ...
```

两个实现：

| | 真实仿真器（由任务提供） | `NeuralSim`（与任务无关） |
| --- | --- | --- |
| 内部 | PushT：`B` 个 `gym-pusht` 环境，逐个步进 | 一个世界模型，`B` 个回合一次批量前向 |
| `reset` | 设定初始状态后渲染首帧 | 调用真实仿真器渲染首帧，编码后把它重复 `K` 次作为初始上下文 |
| `progress` | 仿真器真值 | 判定头从隐状态读出 |
| 确定性 | 由 `InitState` 完全决定 | 由 `InitState` 与模型权重决定；latent 后端无采样噪声 |

`NeuralSim.reset` 借用真实仿真器渲染首帧。这是此类评估的通常做法：首帧来自真实世界，之后全部由模型生成。
这也意味着两种仿真器在第 0 步的画面完全相同，之后的差异全部来自模型。

### 4.3 WorldModel

```python
class WorldModel(Protocol):
    context_frames: int  # K

    def encode(self, frames: Tensor) -> Tensor: ...  # [B, T, H, W, 3] uint8 -> [B, T, *Z]
    def predict(self, latents: Tensor, actions: Tensor) -> Tensor:
        ...
        # [B, K, *Z], [B, K, A] -> 下一步隐状态 [B, *Z]

    def decode(self, latents: Tensor) -> Tensor: ...  # [B, *Z] -> [B, H, W, 3] uint8
    def read_task(self, latents: Tensor) -> dict[str, Tensor]:
        ...
        # [B, *Z] -> {"progress": [B], **任务声明的标签}
```

`*Z` 是后端自己决定的隐状态形状（latent 后端是一张小的特征图）。`NeuralSim` 不关心它的内容，只负责维护一个长度为 `K` 的滚动窗口。
世界模型在构造时接收 `TaskSpec`，据此确定动作维度、画面尺寸和判定头的输出。

第一个后端（`latent`）：卷积自编码器 + 隐空间里的动作条件因果 Transformer，确定性预测。
第二个后端（`diffusion`，P2）：同一个接口，`predict` 内部做少步去噪；此时需要给接口加一个随机数发生器参数。

### 4.4 Policy

```python
class Policy(Protocol):
    name: str
    obs_frames: int  # 需要的历史帧数
    chunk: int  # 输出的动作分块长度 H

    def act(self, frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ...
        # [B, obs_frames, H, W, 3] uint8 -> [B, chunk, A]，环境坐标
```

策略只接收画面。评测器负责维护帧历史、把分块里的前若干个动作依次执行、再重新调用策略。
「执行噪声」这类变体用包装器实现（包一个策略，在输出上加噪声），本身也是一个 `Policy`。

### 4.5 评测循环

```
evaluate(policy, sim, suite, max_steps, n_exec):
    frames ← sim.reset(suite.inits)                       # [B, H, W, 3]
    history ← 把首帧重复 obs_frames 次
    done ← 全 False；best ← 全 0
    当 步数 < max_steps 且 还有未结束的回合：
        chunk ← policy.act(history, rng)                  # [B, chunk, A]
        对 chunk 的前 n_exec 个动作依次：
            result ← sim.step(action)
            best ← max(best, result.progress)；done ← done 或 result.success
            history 滚动一帧
    返回 逐回合记录（是否成功、最大进度、结束步数）
```

已结束的回合继续参与批量前向但结果被屏蔽，这样批的形状不变，实现简单；代价是少量多余计算。

## 5. 数据

### 5.1 回合文件

下表是 PushT 的回合文件。通用字段是 `frames`、`actions`、`progress`、`success`、`meta`；其余是任务声明的标签。

每个回合一个 `.npz`。下标 `t` 处放 $o_t$、$a_t$；`T` 个动作配 `T + 1` 帧画面。

| 字段 | 形状 | 类型 | 说明 |
| --- | --- | --- | --- |
| `frames` | `[T+1, 96, 96, 3]` | uint8 | |
| `actions` | `[T, 2]` | float32 | 环境坐标 0–512 |
| `label__agent_pos` | `[T+1, 2]` | float32 | 推杆位置 |
| `label__block_pose` | `[T+1, 3]` | float32 | 块的 x、y、朝向；人类演示里没有，填 NaN |
| `progress` | `[T+1]` | float32 | 即 `coverage`。人类演示里由奖励还原（奖励 = `clip(coverage / 0.95, 0, 1)`），每个回合的首帧未知，填 NaN |
| `success` | `[T+1]` | bool | |
| `meta` | JSON 字符串 | | 来源（`demo` / `replay` / `random_walk`，M2 之后加 `policy:<名称>`）、序号、所属的组、初始状态、是否带标签 |

人类演示以「T 帧 + T 动作」的形式发布，最后一个动作之后的画面不存在。转换时丢掉最后一个动作，得到 `T−1` 个动作配 `T` 帧。

带标签与不带标签的回合混在一个数据集里：世界模型的预测损失用全部回合，判定头的损失只在带标签的帧上计算。

### 5.2 数据集与评测集

```
workspace/datasets/<任务>/<名称>/
├── episodes/episode-000000.npz …
├── manifest.json          回合清单：文件、长度、来源、所属的组、内容摘要；切分；指纹；动作与标签的归一化统计量
└── REPORT.md              校验报告
workspace/suites/<任务>/<名称>.json   评测集：一组 InitState + 指纹
```

- **指纹**：对每个回合文件里各数组的原始字节求 SHA-256，再对排序后的摘要列表求一次。不直接对 `.npz` 文件求哈希，因为压缩包里含时间戳。
- **切分**：按「组」划分，由种子决定，写在清单里。一条演示和由它加噪重放出来的回合属于同一组，整组进同一个子集，
  否则验证集里会出现和训练集共用同一条动作序列的回合。演示组与其他组分开洗牌，保证每个子集里两类数据都有。
  归一化统计量只用训练部分计算。
- **校验**：生成时逐回合检查形状与类型、动作范围、数值范围、标签完整；仿真器内采集的回合每种来源抽查若干个做重放
  （从记录的初始状态重新执行动作，画面逐像素相同）。任何一项不通过，数据集不会写出清单。
- **评测集**：初始状态由独立的种子序列生成，与数据采集用的种子不相交。

### 5.3 训练采样

序列窗口采样：从一个回合里取连续的 `L` 步，不跨回合。标准规模下全部画面常驻内存
（人类演示的画面约 700 MB，已实测）；数据量超过内存上限时改为按回合懒加载，接口不变。

## 6. 存储

### 6.1 工作区布局

```
workspace/
├── roboarena.db
├── datasets/  suites/
├── runs/<run_id>/
│   ├── config.toml         生效的完整配置
│   ├── meta.json           代码版本、是否有未提交改动、数据指纹、设备、起止时间
│   ├── metrics.jsonl       每行一条：步数、指标名、值
│   ├── checkpoints/        last.pt（续训用，含优化器与随机数状态）、best.pt
│   └── log.txt
├── artifacts/<sha256 前缀>/  已注册的检查点（只含权重与构造参数）
└── reports/                生成的图表
```

`run_id` 形如 `20261125-213005-wm-latent-a1b2`（时间 + 类型 + 配置摘要的前几位）。

### 6.2 数据库表

用 SQLite，标准库 `sqlite3`，不引入 ORM。

| 表 | 主要字段 | 说明 |
| --- | --- | --- |
| `runs` | `id`、`kind`（`data` / `train_policy` / `train_wm` / `eval`）、`status`、`config_json`、`git_rev`、`git_dirty`、`dataset_fp`、`seed`、`started_at`、`finished_at` | 每次命令行或任务执行一行 |
| `artifacts` | `id`、`task`、`name`（唯一）、`kind`（`policy` / `worldmodel`）、`backend`、`sha256`、`path`、`run_id`、`params_json`、`created_at` | 注册表 |
| `suites` | `id`、`task`、`name`、`fingerprint`、`n_inits`、`path` | 评测集 |
| `evals` | `id`、`policy_id`、`sim`（`real` 或世界模型的 `artifact id`）、`suite_id`、`max_steps`、`seed`、`run_id`、`status` | 一次评测；前六个字段联合唯一 |
| `episodes` | `eval_id`、`init_index`、`success`、`max_progress`、`steps` | 逐回合结果，统计量全部由它现算 |
| `jobs` | `id`、`kind`、`payload_json`、`status`、`progress`、`error`、`trace_id`、`token_id`、`created_at`、`updated_at` | 服务提交的任务 |
| `tokens` | `id`、`name`、`hash`、`role`（`read` / `write`）、`created_at`、`revoked_at` | API 令牌，只存哈希 |

一次评测的策略、仿真器、评测集必须属于同一个任务，写入时检查。

排行榜、秩相关等汇总结果**不落表**，每次由 `episodes` 现算。这样统计方法改了不需要迁移数据，也不会出现汇总与明细不一致。

### 6.3 检查点的完整性

注册时计算文件的 SHA-256 并按内容寻址存放；加载时重新核对。
加载一律用 `torch.load(..., weights_only=True)`，构造参数存在数据库的 `params_json` 里，不从检查点里反序列化任意对象。

## 7. 配置

三层，后者覆盖前者：代码里的 dataclass 默认值 → TOML 文件 → 命令行的 `键=值`。

```toml
# configs/local.toml（结构示意，数值在实现时确定）
[workspace]
root = "workspace"

[task]
name = "pusht"

[data]
n_replay = …
n_random_walk = …

[worldmodel]
backend = "latent"
context_frames = …

[policy.bc]
obs_frames = …
chunk = …

[eval]
suite = "main"
max_steps = 300
n_exec = …
```

用标准库 `tomllib` 读取。未知的键直接报错，避免拼错的配置项被悄悄忽略。
环境变量 `ROBOARENA_QUICK=1` 等价于选用 `quick.toml`。

## 8. 命令行

```
roboarena data build            下载演示、采集、校验、切分、写清单
roboarena data info             显示数据集的校验报告
roboarena data verify           重新读取全部回合文件，核对指纹并重跑检查
roboarena data suite            生成评测集
roboarena policy add|list       登记不需要训练的策略；列出策略
roboarena runs list             最近的运行记录
roboarena task list             已注册的任务
roboarena info                  设备、工作区概况
roboarena train policy          训练一个策略并注册
roboarena train worldmodel      训练一个世界模型并注册
roboarena eval run              在指定仿真器上评估指定策略（可一次指定多个）
roboarena eval agreement        计算一致性指标
roboarena eval openloop         开环误差曲线与可信视野
roboarena report                生成图表与结果表
roboarena bench                 延迟基准
roboarena serve                 启动服务
roboarena token create|list|revoke    管理 API 令牌
roboarena pipeline              按配置依次执行以上全部离线步骤
```

每个子命令都接受 `--config` 和若干 `--set 键=值` 覆盖。`pipeline` 是幂等的：已完成且输入指纹未变的步骤会跳过。

## 9. 服务

### 9.1 REST 接口

前缀 `/api`。请求与响应用 pydantic 模型定义，文档由 FastAPI 自动生成。

| 方法与路径 | 作用 |
| --- | --- |
| `GET /api/health/live` | 存活：进程在响应 |
| `GET /api/health/ready` | 就绪：数据库可达、任务执行器在运行；附版本与设备 |
| `GET /metrics` | Prometheus 文本格式的指标 |
| `GET /api/tasks` | 已注册的任务及其规格 |
| `GET /api/artifacts?task=&kind=` | 策略与世界模型列表 |
| `GET /api/suites?task=` | 评测集列表 |
| `GET /api/leaderboard?suite=&sim=` | 各策略的成功率与 Wilson 区间 |
| `GET /api/agreement?suite=&worldmodel=` | 秩相关及区间、MMRV、偏差、散点数据 |
| `GET /api/openloop?worldmodel=` | 开环误差曲线、基线曲线、可信视野 |
| `GET /api/runs`、`GET /api/runs/{id}` | 运行记录与指标曲线 |
| `GET /api/evals/{id}/episodes` | 某次评测的逐回合结果 |
| `POST /api/jobs` | 提交评测任务 |
| `GET /api/jobs`、`GET /api/jobs/{id}` | 任务状态与进度 |

所有引用都用数据库里的名称或编号，接口不接受文件路径。
`GET` 接口需要只读权限，`POST /api/jobs` 与遥操作需要可写权限；健康检查与 `/metrics` 的鉴权见 9.5。

### 9.2 任务执行器

一个后台线程，从 `jobs` 表里按提交顺序取任务，串行执行。理由：只有一块 GPU，并行评测只会互相争抢。
任务进度定期写回 `jobs.progress`。服务启动时，把上次遗留的「运行中」任务标为「已中断」。

### 9.3 遥操作协议

`WebSocket /ws/teleop`。控制消息用 JSON 文本帧，画面用二进制帧。

```
客户端 → 服务端
  {"type": "start", "worldmodel": "<名称>", "compare_real": true, "init": <评测集里的下标 或 null>}
  {"type": "action", "a": [<A 个数，环境坐标>], "seq": <递增整数>}
  {"type": "reset"}

服务端 → 客户端
  {"type": "ready", "session": "...", "hz": <任务的控制频率>, "frame": [H, W], "action_low": […], "action_high": […]}
  {"type": "frame", "seq": …, "step": …, "progress": …, "real_progress": … 或 null,
   "ms": {"model": …, "encode": …, "total": …}}          紧跟一个二进制帧
  二进制帧：1 字节通道号（0 神经仿真器，1 真实仿真器）+ PNG 编码的画面
  {"type": "error", "message": "..."}
```

节拍由服务端掌握：按任务的控制周期（PushT 为 100 毫秒）取客户端最近一次发来的动作步进一次。客户端发得再快也不会让仿真加速，
发得慢则沿用上一个动作。这样仿真速度与真实任务一致，延迟也好测。
画面放大由浏览器完成（canvas 关闭平滑，保留像素感）。

模型推理在工作线程里执行，不阻塞事件循环。同时在线的会话数有上限（默认 2），超出则拒绝。
把鼠标位置换算成动作是前端的事：PushT 的动作恰好是画面上的一个点，按 `action_low`、`action_high` 线性映射即可。

### 9.4 前端

React + Vite + TypeScript 的单页应用。开发时由 Vite 的开发服务器提供页面并把 `/api`、`/ws` 代理到后端；
发布时 `vite build` 的产物放在 `web/dist/`，由 FastAPI 作为静态文件托管，整个系统仍然只有一个进程、一个端口。

| 层 | 做法 |
| --- | --- |
| 路由 | 三个页面：排行榜、一致性、遥操作 |
| 接口类型 | 由后端的 OpenAPI 描述生成 TypeScript 类型，生成结果入库；持续集成里重新生成并比对，不一致则失败 |
| 数据获取 | 一层薄的请求封装（带令牌、统一错误处理）+ 每个接口一个钩子 |
| 图表 | 用 SVG 写成 React 组件：带区间的条形图、带误差线的散点图、折线图。只有这三种，不引入图表库 |
| 遥操作 | 一个管理 WebSocket 的钩子（连接、重连、最近一帧、延迟统计）+ 一个 canvas 组件（关闭平滑，保留像素感） |
| 状态 | 页面内的局部状态即可，不引入全局状态库 |
| 测试 | Vitest：数据变换、图表的坐标计算、WebSocket 钩子的消息处理 |

| 页面 | 内容 |
| --- | --- |
| 排行榜 | 表格 + 区间条形图；可切换任务、评测集与仿真器 |
| 一致性 | 真实对神经的散点图（带区间的误差线）、秩相关与区间、MMRV、开环误差曲线 |
| 遥操作 | 画布（神经仿真器，可并排真实仿真器）、实时帧率与延迟、重置、选择初始状态 |

前端依赖的具体版本在建工程时（M5b）按当时的稳定版确定，写进锁文件。

### 9.5 鉴权与限流

- **令牌**：`roboarena token create --name … --role read|write` 生成一个随机令牌，明文只显示这一次，数据库里存 SHA-256。
  令牌本身是高熵随机串，所以用普通哈希即可，不需要口令哈希算法。
- **传递**：HTTP 用 `Authorization: Bearer <令牌>`。WebSocket 在握手时校验；浏览器的 WebSocket 接口不能自定义请求头，
  因此前端把令牌放在子协议字段里传，不放在地址里（地址会进访问日志）。
- **检查**：FastAPI 的依赖项。`require_read`、`require_write` 两个依赖挂在各自的路由上；失败返回 401（没有或无效）或 403（角色不够）。
- **何时强制**：配置项 `serve.auth` 取 `off`、`on`。监听地址不是回环地址而 `auth = off` 时，服务拒绝启动。
- **健康检查与指标**：`/api/health/live` 始终免鉴权（给容器探针用）；`/api/health/ready` 与 `/metrics` 在启用鉴权时需要只读令牌。
- **限流**：进程内实现。遥操作按在线会话数限制；`POST /api/jobs` 按令牌做滑动窗口计数。超出返回 429。
  单进程部署下进程内计数就是准确的，不需要外部存储。

### 9.6 可观测性

- **日志**：标准库 `logging` 加一个 JSON 格式化器，写到标准输出。字段：`ts`、`level`、`logger`、`msg`、`trace_id`，以及各处附加的键值。
  令牌与请求头里的 `Authorization` 不进日志。命令行的训练日志沿用同一套格式，同时写进运行目录的 `log.txt`。
- **追踪编号**：中间件为每个请求生成一个（或沿用请求头 `X-Request-ID`），写回响应头，放进上下文变量供日志取用。
  提交任务时把它存进 `jobs.trace_id`，任务执行器处理该任务时恢复到上下文里，这样一次点击触发的全部日志能串起来。
- **指标**：

| 指标 | 类型 | 标签 |
| --- | --- | --- |
| `roboarena_http_requests_total` | 计数 | 方法、路由、状态码 |
| `roboarena_http_request_seconds` | 直方图 | 方法、路由 |
| `roboarena_inference_seconds` | 直方图 | 环节（编码、预测、解码、判定）、后端 |
| `roboarena_teleop_frame_seconds` | 直方图 | — |
| `roboarena_teleop_sessions` | 瞬时值 | — |
| `roboarena_jobs_queued`、`roboarena_jobs_running` | 瞬时值 | — |
| `roboarena_jobs_total` | 计数 | 类型、结果 |

  路由标签用路由模板而不是实际路径，避免标签取值无限增长。

## 10. 部署形态

| 形态 | 用途 | 设备 |
| --- | --- | --- |
| 本机 | 开发、标准规模训练、服务与演示 | MPS |
| 云端 4090 | 放大规模训练（`cloud.toml`）、扩散后端 | CUDA |
| Docker | 只跑服务：挂载一个已有的工作区，提供看板与遥操作 | CPU |

容器镜像不包含训练数据与权重，工作区通过卷挂载。镜像分两个阶段构建：先用 Node 构建前端，再把产物拷进 Python 运行镜像，最终镜像里没有 Node。
镜像里的服务监听 `0.0.0.0`（容器内），因此默认启用鉴权，令牌通过挂载的工作区里的数据库管理；
直接运行时默认监听 `127.0.0.1`。

### 10.1 持续集成

GitHub Actions，四个作业：

| 作业 | 内容 |
| --- | --- |
| `backend` | `ruff`、静态类型检查、除端到端以外的 `pytest`（CPU，不联网） |
| `frontend` | 代码检查、`tsc --strict`、Vitest、`vite build`；重新生成接口类型并与入库的比对 |
| `e2e-smoke` | 用玩具任务在冒烟配置下跑 `roboarena pipeline`，再启动服务查一次排行榜接口 |
| `image` | 构建镜像，启动容器并通过存活检查；不推送 |

端到端作业用玩具任务而不是 PushT：不需要下载演示数据，也不依赖渲染相关的系统库，跑得快且稳定。
PushT 的全流程冒烟在本机手动跑。

## 11. 可复现性

- 每次运行记录：完整配置、代码版本与是否有未提交改动、数据指纹、种子、设备、依赖锁文件的摘要。
- 随机数：每个组件用独立的发生器，由（全局种子，组件名）派生，避免「多采一个随机数，后面全变」。
- 评测的确定性：`RealSim` 由初始状态决定；`NeuralSim`（latent 后端）无采样；策略的随机性来自传入的发生器。
- 续训用的检查点保存模型、优化器、步数、随机数状态与数据采样位置。
- MPS 上部分算子不保证逐位确定。要求逐位一致的测试在 CPU 上跑；MPS 上只要求指标在容差之内。

## 12. 测试策略

| 层 | 测什么 | 例子 |
| --- | --- | --- |
| 单元 | 纯函数 | Wilson 区间、Spearman、MMRV 对手算例子；指纹对内容敏感、对文件时间戳不敏感 |
| 数据 | 校验规则 | 把动作错开一步，校验能发现；切分没有回合泄漏 |
| 接口契约 | 两种仿真器遵守同一份约定 | 同一组契约测试对 `RealSim` 和 `NeuralSim` 各跑一遍：形状、类型、首帧相同、批量与逐个一致 |
| 模型 | 能学习 | 在一个批上过拟合，损失明显下降；续训与不中断训练等价 |
| 评测 | 评测器的逻辑 | 用一个脚本化的假策略和假仿真器，成功率是可预先算出的 |
| 任务契约 | 每个任务插件遵守同一份约定 | 同一组测试对 `pusht` 和 `toy` 各跑一遍：采样初始状态可序列化往返、真实仿真器确定、规格与实际输出相符 |
| 服务 | HTTP 与 WebSocket | 用测试客户端走一遍查询、提交任务、遥操作会话 |
| 鉴权 | 访问控制 | 无令牌 401、只读令牌提交任务 403、吊销后失效、对外监听而未启用鉴权时拒绝启动、日志里没有令牌 |
| 可观测性 | 日志与指标 | 响应头带追踪编号且与日志一致；请求后对应的计数增加 |
| 前端 | 组件与数据逻辑 | 图表坐标计算、区间渲染、WebSocket 消息处理 |
| 端到端 | 全流程 | 冒烟配置下 `pipeline` 跑通，报告文件生成（玩具任务进持续集成，PushT 在本机跑） |
| 架构 | 依赖方向 | 扫描导入语句，违反第 3 节的规则则失败；在 `tasks/pusht/` 以外搜到 `gym_pusht` 或 `coverage` 则失败 |

测试默认在 CPU 上跑、不访问网络；需要人类演示的测试用一个随仓库提供的极小样本或标记为可跳过。
持续集成的作业划分见 10.1。

## 13. 依赖

| 包 | 用途 |
| --- | --- |
| `torch`、`numpy` | 模型与数值 |
| `gym-pusht`（带 `gymnasium`、`pymunk`、`pygame-ce`） | 真实仿真器 |
| `pyarrow`、`opencv-python-headless` | 读取演示数据的表格与视频；PNG 编码 |
| `matplotlib` | 报告里的图 |
| `fastapi`、`uvicorn[standard]` | 服务与 WebSocket |
| `prometheus-client` | 指标端点 |
| 开发：`pytest`、`httpx`、`ruff`、`pyright` | 测试与检查 |
| 前端：`react`、`react-router`、`vite`、`typescript`、`vitest`、`openapi-typescript` | 页面、构建、测试、接口类型生成 |

`gym-pusht` 声明依赖 `pygame` 与带界面的 `opencv-python`。沿用课程仓库已验证的做法：
在 `[tool.uv] override-dependencies` 里把这两个排除，改用 `pygame-ce` 与 headless 版 OpenCV。
演示数据的下载与解码逻辑从课程仓库的 `src/utils/pusht.py` 移植过来（那份代码已实际跑通），不安装 `lerobot` 本体。

## 14. 关键设计决策

| 决策 | 选择 | 放弃的方案 | 理由 |
| --- | --- | --- | --- |
| 策略的输入 | 只有画面 | 画面 + 推杆坐标 | 神经仿真器只生成画面。若策略还要坐标，就得从画面里再估一次，引入第二个误差来源，结论会说不清 |
| 第一个世界模型后端 | 隐空间的确定性预测 | 像素空间的扩散模型 | 本机能训、单步推理快、评测确定可复现。扩散后端作为可插拔的扩展，用来回答「更强的生成模型是否让评估更可信」 |
| 成功判定 | 从隐状态读出的判定头 | 在生成的画面上跑检测 | 最简单，且可以单独在真实画面上检验。画面检测作为消融里的对照 |
| 神经仿真器的首帧 | 由真实仿真器渲染 | 由模型生成 | 两种仿真器从完全相同的画面出发，差异全部归因于模型 |
| 存储 | SQLite + 文件 | 实验跟踪平台、对象存储 | 单机单人，零运维；表结构清楚，换成别的数据库只需改 `store` |
| 汇总结果 | 由逐回合记录现算 | 存进表里 | 统计方法会改；明细是唯一的事实来源 |
| 任务执行 | 进程内单线程队列 | Celery、Redis | 只有一块 GPU，串行就是正确的调度；少两个外部组件 |
| 遥操作的节拍 | 服务端定时 | 客户端每发一个动作步进一次 | 仿真速度与真实任务一致；网络抖动不改变仿真的时间轴 |
| 前端 | React + Vite + TypeScript | 原生 ES 模块，无构建 | 评审时确定：更接近正式产品的做法，类型能和后端接口对齐。工作量更大，靠「M4 之后才建前端工程」来控制 |
| 图表 | 手写 SVG 组件 | 图表库 | 只需要三种图，且都要画置信区间；自己画比配置图表库更直接，也少一个依赖 |
| 任务抽象 | 现在就做成插件，配一个玩具任务 | 等第二个任务出现再抽象 | 评审时确定。接口形状事后改代价大；玩具任务让抽象有第二个使用者，不至于凭空设计 |
| 鉴权 | 静态 API 令牌 + 两种角色 | OAuth、会话登录 | 使用者是脚本和少数人；令牌足够，且不需要账号体系 |
| 令牌存储 | SHA-256 | bcrypt、argon2 | 令牌是高熵随机串，不存在撞库与弱口令问题；慢哈希只会拖慢每个请求 |
| 限流 | 进程内计数 | Redis | 单进程部署，进程内就是准确的 |
| 指标 | Prometheus 文本端点 | 推送到某个监控平台 | 拉取式端点不绑定任何后端；本机用 `curl` 就能看 |
| 配置 | dataclass + TOML | Hydra、OmegaConf | 标准库即可；未知键报错这一条自己写更可控 |
| 包结构 | 可安装的 `src/` 布局 | 课程仓库式的脚本目录 | 独立软件，测试与服务需要正常导入 |

## 15. 扩展点

- **新策略**：实现 `Policy`，注册到 `policies` 的工厂表，训练后登记为 `artifact`。
- **新世界模型后端**：实现 `WorldModel`，注册到 `worldmodel` 的工厂表。评测、服务、前端不需要改。
- **新任务**：在 `tasks/` 下实现 `Task`（规格、初始状态采样、真实仿真器），注册名称。
  数据采集的行为策略（加噪回放、随机游走）里与任务相关的部分也放在任务插件里。评测、服务、前端不需要改。
