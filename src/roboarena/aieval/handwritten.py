"""人工编写的指令（docs/PLAN.md 1.1 节的 handwritten 桶）：不受模板的措辞限制。

状态：Claude 起草的初稿，**尚待用户审改**；按计划在微调（A3）开始之前冻结，冻结之后不再改动。
这些指令不进微调语料（A3 的语料测试按规范化后的文本比对）。

每条：（用例族，指令，工具，标准参数）。标准参数只写用户明确说到的，其余的由默认值补齐。
"""

from __future__ import annotations

from typing import Any

HANDWRITTEN: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
    # ---- 列出策略 ----
    ("list_policies", "咱们现在都有哪些策略啊", "list_policies", {}),
    ("list_policies", "show me all policies", "list_policies", {}),
    ("list_policies", "策略清单发我一份", "list_policies", {}),
    ("list_policies", "我忘了都训过哪些策略了，帮我看看", "list_policies", {}),
    ("list_policies", "有几个策略登记在册？列出来", "list_policies", {}),
    # ---- 排行榜（不指定评测集）----
    ("leaderboard_default", "排行榜", "leaderboard", {}),
    ("leaderboard_default", "把榜单拉出来我看看", "leaderboard", {}),
    ("leaderboard_default", "各策略成功率汇总一下", "leaderboard", {}),
    ("leaderboard_default", "leaderboard 看一眼", "leaderboard", {}),
    ("leaderboard_default", "当前评测集上大家的成绩如何", "leaderboard", {}),
    # ---- 排行榜（指定评测集）----
    ("leaderboard", "probe 的榜单", "leaderboard", {"suite": "probe"}),
    ("leaderboard", "bench 上大家成绩怎么样", "leaderboard", {"suite": "bench"}),
    ("leaderboard", "想看看 probe 这个评测集的结果汇总", "leaderboard", {"suite": "probe"}),
    ("leaderboard", "bench 评测集，成功率从高到低排一下", "leaderboard", {"suite": "bench"}),
    ("leaderboard", "切到 probe 看排名", "leaderboard", {"suite": "probe"}),
    ("leaderboard", "leaderboard for suite bench", "leaderboard", {"suite": "bench"}),
    # ---- 排行榜的前几名 ----
    ("leaderboard_top", "bench 上的冠军是谁，只要第一名", "leaderboard", {"suite": "bench", "top": 1}),
    ("leaderboard_top", "probe 排前两名的策略", "leaderboard", {"suite": "probe", "top": 2}),
    ("leaderboard_top", "bench 榜单 top 3", "leaderboard", {"suite": "bench", "top": 3}),
    ("leaderboard_top", "只给我看 bench 前五", "leaderboard", {"suite": "bench", "top": 5}),
    ("leaderboard_top", "排行榜前三名", "leaderboard", {"top": 3}),
    # ---- 比较两个策略 ----
    ("compare", "atlas 和 random 谁厉害？在 bench 上比", "compare_policies", {"a": "atlas", "b": "random", "suite": "bench"}),
    ("compare", "atlas-noise0.3 比 atlas 好吗，用 probe 的结果", "compare_policies", {"a": "atlas-noise0.3", "b": "atlas", "suite": "probe"}),
    ("compare", "对比一下 atlas-step20 跟 atlas-step60", "compare_policies", {"a": "atlas-step20", "b": "atlas-step60"}),
    ("compare", "bench 上 atlas-demos2 和 atlas 的差距有统计意义吗", "compare_policies", {"a": "atlas-demos2", "b": "atlas", "suite": "bench"}),
    ("compare", "random vs atlas-noise0.1 on bench", "compare_policies", {"a": "random", "b": "atlas-noise0.1", "suite": "bench"}),
    ("compare", "多训 40 步有用吗？比较 atlas-step20 和 atlas-step60，评测集 bench", "compare_policies", {"a": "atlas-step20", "b": "atlas-step60", "suite": "bench"}),
    ("compare", "atlas-noise0.1 和 atlas-noise0.3 这两个加了噪声的版本，哪个成功率高", "compare_policies", {"a": "atlas-noise0.1", "b": "atlas-noise0.3"}),
    # ---- 运行记录 ----
    ("list_runs", "最近跑了些什么", "list_runs", {}),
    ("list_runs", "最近 5 次训练", "list_runs", {"kind": "train_policy", "limit": 5}),
    ("list_runs", "评测类的运行，最新 3 条", "list_runs", {"kind": "eval", "limit": 3}),
    ("list_runs", "看看最近的数据构建记录", "list_runs", {"kind": "data"}),
    ("list_runs", "last 20 runs", "list_runs", {"limit": 20}),
    ("list_runs", "最近一次训练是哪次运行", "list_runs", {"kind": "train_policy", "limit": 1}),
    # ---- 某一次运行 ----
    ("get_run", "20261127-140257-train_policy-1ab7df54 这次跑成功了没", "get_run", {"run_id": "20261127-140257-train_policy-1ab7df54"}),
    ("get_run", "运行 20261126-093015-eval-77c2e0aa 的指标", "get_run", {"run_id": "20261126-093015-eval-77c2e0aa"}),
    ("get_run", "帮我看下 20261125-221904-data-03bd91f6", "get_run", {"run_id": "20261125-221904-data-03bd91f6"}),
    ("get_run", "20261127-081233-eval-5e1a9c40 是用哪个策略跑的", "get_run", {"run_id": "20261127-081233-eval-5e1a9c40"}),
    ("get_run", "编号 20261124-170045-train_policy-c4d2a811，验证损失最后是多少", "get_run", {"run_id": "20261124-170045-train_policy-c4d2a811"}),
    # ---- 数据集 ----
    ("dataset_info", "core 数据集多大", "dataset_info", {"name": "core"}),
    ("dataset_info", "mini 里有几条回合", "dataset_info", {"name": "mini"}),
    ("dataset_info", "数据集概况", "dataset_info", {}),
    ("dataset_info", "core 的训练集和验证集各多少回合", "dataset_info", {"name": "core"}),
    ("dataset_info", "当前数据集的指纹是什么", "dataset_info", {}),
    # ---- 配置说明 ----
    ("describe_config", "policy 段能调哪些参数", "describe_config", {"section": "policy"}),
    ("describe_config", "eval 那段配置都有啥", "describe_config", {"section": "eval"}),
    ("describe_config", "所有配置项列一下", "describe_config", {}),
    ("describe_config", "diffusion 段有哪些设置", "describe_config", {"section": "diffusion"}),
    ("describe_config", "data 段里的 n_replay 是什么意思", "describe_config", {"section": "data"}),
    # ---- 校验覆盖项 ----
    ("check_overrides", "policy.lr=0.01 这个覆盖项能用吗", "check_overrides", {"overrides": ["policy.lr=0.01"]}),
    ("check_overrides", "我想设 eval.max_steps=100 和 eval.batch=8，先帮我确认写法没问题", "check_overrides", {"overrides": ["eval.max_steps=100", "eval.batch=8"]}),
    ("check_overrides", "policy.stpes=1000 对吗", "check_overrides", {"overrides": ["policy.stpes=1000"]}),
    ("check_overrides", "验证一下：diffusion.sample_steps=20", "check_overrides", {"overrides": ["diffusion.sample_steps=20"]}),
    ("check_overrides", "policy 段，学习率 0.0005、批大小 64，写成覆盖项检查一下，别真跑", "check_overrides", {"overrides": ["policy.lr=0.0005", "policy.batch=64"]}),
    # ---- 评测 ----
    ("run_eval", "atlas-demos2 在 probe 上的成绩还没有，跑一个", "run_eval", {"policy": "atlas-demos2", "suite": "probe"}),
    ("run_eval", "评测 atlas", "run_eval", {"policy": "atlas"}),
    ("run_eval", "把 random 在 bench 上重新评一遍，覆盖掉旧结果", "run_eval", {"policy": "random", "suite": "bench", "force": True}),
    ("run_eval", "run eval for atlas-step60 on probe", "run_eval", {"policy": "atlas-step60", "suite": "probe"}),
    ("run_eval", "给 atlas-noise0.1 跑个评测，评测集 probe", "run_eval", {"policy": "atlas-noise0.1", "suite": "probe"}),
    ("run_eval", "atlas-step20 重新评测，旧的结果不要了", "run_eval", {"policy": "atlas-step20", "force": True}),
    ("run_eval", "用 bench 测一下 atlas-noise0.3", "run_eval", {"policy": "atlas-noise0.3", "suite": "bench"}),
    # ---- 训练 ----
    ("train_policy", "训一个叫 comet 的策略", "train_policy", {"name": "comet"}),
    ("train_policy", "用 diffusion 训练 drift，2000 步", "train_policy", {"name": "drift", "backend": "diffusion", "overrides": ["diffusion.steps=2000"]}),
    ("train_policy", "新策略 ember：行为克隆，学习率 1e-4，训练 800 步", "train_policy", {"name": "ember", "backend": "bc", "overrides": ["policy.lr=1e-4", "policy.steps=800"]}),
    ("train_policy", "只用 20 条演示训练一个 bc 策略，名字 flint", "train_policy", {"name": "flint", "backend": "bc", "overrides": ["policy.n_demos=20"]}),
    ("train_policy", "train a diffusion policy named glint with batch size 32", "train_policy", {"name": "glint", "backend": "diffusion", "overrides": ["diffusion.batch=32"]}),
    ("train_policy", "再训一个行为克隆，叫 haze，其他都用默认", "train_policy", {"name": "haze", "backend": "bc"}),
    # ---- 删除 ----
    ("remove_policy", "random 这个策略没用了，删了吧", "remove_policy", {"name": "random"}),
    ("remove_policy", "delete atlas-step20", "remove_policy", {"name": "atlas-step20"}),
    ("remove_policy", "把 atlas-noise0.3 从库里拿掉", "remove_policy", {"name": "atlas-noise0.3"}),
    ("remove_policy", "atlas-demos2 可以删了", "remove_policy", {"name": "atlas-demos2"}),
    ("remove_policy", "清理掉 atlas-step60，连同评测结果", "remove_policy", {"name": "atlas-step60"}),
)  # fmt: skip
