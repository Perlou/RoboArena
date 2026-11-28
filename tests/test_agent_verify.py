"""结果校验：回答里的数字必须能在本次会话的工具输出或用户输入里找到出处（docs/PLAN.md 1.2 节）。

「什么算数字、什么算有出处」的规则就写在这些测试里。校验只管数字，不管文字性的错误。
"""

import pytest

from roboarena.agent.verify import mentions, verify

BOARD = '{"rows": [{"rank": 1, "policy": "atlas-noise0.3", "successes": 9, "n": 16, "success_rate": 0.5625, "ci95": [0.3318, 0.769]}]}'


def _unsupported(answer: str, *sources: str) -> list[str]:
    return [m.text for m in verify(answer, sources).unsupported]


# ---- 有出处 ----


@pytest.mark.parametrize(
    "answer",
    [
        "成功率是 0.5625。",
        "成功 9 次，共 16 回合。",
        "成功率 9/16。",
        "成功率约 0.56。",  # 四舍五入到回答里写出的位数
        "成功率约 0.563。",
        "成功率是 56.25%。",  # 小数换成百分比
        "成功率约 56%。",
        "成功率约 56.3％。",  # 全角的百分号
        "区间是 [0.33, 0.77]。",
        "95% 区间的上界是 0.769。",  # 95 出自字段名 ci95：出处里连在名称里的数字也算
        "排第 1 名的是 atlas-noise0.3。",  # 名称里的 0.3 在工具输出里出现过
    ],
)
def test_numbers_from_tool_output_are_supported(answer):
    assert _unsupported(answer, BOARD) == []


def test_numbers_from_the_user_are_supported():
    assert _unsupported("好的，按你说的训练 500 步、学习率 0.001。", "训练 500 步，学习率 1e-3") == []
    assert _unsupported("阈值 0.9 对应 90%。", "阈值设成 90%") == []  # 用户写百分比，回答写小数，或照抄


def test_thousands_separators_and_scientific_notation():
    assert _unsupported("一共 92,719 帧。", '{"n_frames": 92719}') == []
    assert _unsupported("p 值是 3.2e-20。", '{"mcnemar_p": 3.208e-20}') == []
    assert _unsupported("p 值约为 0.000017。", '{"mcnemar_p": 1.705e-05}') == []
    assert _unsupported("p 值是 3.9e-20。", '{"mcnemar_p": 3.208e-20}') == ["3.9e-20"]


def test_sign_is_ignored():
    """只比大小不比正负：「下降了 0.2」和 -0.2 是同一个数。正负写错属于文字性的错误，不在校验范围内。"""
    assert _unsupported("损失下降了 0.2。", '{"delta": -0.2}') == []


# ---- 没有出处 ----


@pytest.mark.parametrize(
    "answer, bad",
    [
        ("成功率是 0.61。", ["0.61"]),
        ("成功率是 0.57。", ["0.57"]),  # 0.5625 四舍五入到两位是 0.56，不是 0.57
        ("成功率是 0.55。", ["0.55"]),  # 截断也不行
        ("成功 9 次，共 20 回合。", ["20"]),
        ("成功率 75%。", ["75%"]),
        ("成功率 0.5625，比基线高 0.1。", ["0.1"]),  # 自己算出来的差值没有出处：照定义算没有
        ("一共 3 个策略。", ["3"]),
    ],
)
def test_made_up_numbers_are_found(answer, bad):
    assert _unsupported(answer, BOARD) == bad


def test_rounding_cannot_swallow_the_whole_value():
    """四舍五入只能丢掉比写出来的那一位更小的部分：0.5625 不能说成 1，0.004 不能说成 0.00。"""
    assert _unsupported("成功了 1 次。", '{"success_rate": 0.5625}') == ["1"]
    assert _unsupported("成功率是 1.0。", '{"success_rate": 0.96}') == []  # 0.96 → 1.0 保住了一位有效数字
    assert _unsupported("成功率是 1。", '{"success_rate": 0.96}') == ["1"]
    assert _unsupported("p 值是 0.00。", '{"p": 0.004}') == ["0.00"]
    assert _unsupported("大约 46 分。", '{"score": 46.3}') == []
    assert _unsupported("成功率 0.6。", BOARD) == []  # 0.5625 → 0.6


def test_percent_and_plain_are_not_confused_beyond_the_factor_of_100():
    assert _unsupported("成功率 0.5625%。", BOARD) == []  # 写法怪，但数字本身在出处里
    assert _unsupported("一共 1600 回合。", BOARD) == [
        "1600"
    ]  # 16 不能乘 100 变成 1600：只有带百分号的才换算
    assert _unsupported("比例是 0.16。", BOARD) == ["0.16"]  # 出处里的 16 不是百分比，不能除以 100
    assert _unsupported("比例是 0.16。", "比例定为 16%") == []


def test_only_this_sessions_material_counts():
    assert _unsupported("成功率是 0.5625。") == ["0.5625"]
    assert _unsupported("成功率是 0.5625。", "成功率是多少？") == ["0.5625"]


# ---- 不携带事实的数字不计 ----


def test_list_numbering_is_not_a_claim():
    answer = "结论如下：\n1. atlas-noise0.3 最高\n2. 成功率 0.5625\n3) 共 16 回合\n（4）没有别的了\n- 5、以上"
    assert [m.text for m in mentions(answer)] == ["0.5625", "16"]
    assert _unsupported(answer, BOARD) == []


def test_enumerating_ordinals_are_not_claims_but_ranks_are():
    assert [m.text for m in mentions("第 1 步先看排行榜，第 2 点是区间。")] == []
    assert [m.text for m in mentions("它排在第 3 名。")] == ["3"]  # 名次是事实，要有出处
    assert _unsupported("它排在第 3 名。", BOARD) == ["3"]


def test_digits_inside_names_are_not_numbers():
    """紧跟在英文字母后面的数字是名称的一部分（策略名、模型名），不是数量。"""
    assert [
        m.text for m in mentions("策略 bc-step10000、atlas-noise0.3 和模型 qwen08b，取 top3，版本 v2")
    ] == []
    # 编造的名称属于文字性的错误，不在校验范围内
    assert _unsupported("最好的是 bc-step9999。", BOARD) == []
    # 数字后面跟着单位的照常算
    assert [m.text for m in mentions("耗时 500ms，内存 16GB，7B 的模型")] == ["500", "16", "7"]
    # 运行编号开头的日期与时间不跟在字母后面，照常要有出处
    assert _unsupported("运行 20260101-000000-eval-00000000 已完成。", BOARD) == [
        "20260101",
        "000000",
        "00000000",
    ]


def test_markdown_headings_and_table_rules_do_not_count():
    assert [m.text for m in mentions("## 1. 概况\n| --- | --- |\n### 2）细节")] == []


# ---- 取数的细节 ----


def test_mentions_carry_value_precision_and_percent():
    got = {m.text: (m.value, m.unit, m.percent) for m in mentions("0.460、46%、92,719、3.2e-20、12.50％")}
    assert got == {
        "0.460": (0.46, 0.001, False),
        "46%": (46.0, 1.0, True),
        "92,719": (92719.0, 1.0, False),
        "3.2e-20": (pytest.approx(3.2e-20), pytest.approx(1e-21), False),
        "12.50％": (12.5, 0.01, True),
    }


def test_hex_like_identifiers_are_not_read_as_scientific_notation():
    run_id = "20261127-140257-eval-3e8a1f02"
    assert _unsupported(f"运行 {run_id} 已完成。", f'{{"id": "{run_id}"}}') == []
    assert all(m.value < 1e9 for m in mentions(f"运行 {run_id}"))


def test_a_comma_list_is_not_one_big_number():
    """「16, 160」是两个数。只有三位一组、紧挨着的写法才当作千位分隔。"""
    assert [m.text for m in mentions("分别是 16, 160 和 8,16")] == ["16", "160", "8", "16"]
    # 紧凑的 JSON 里 [16,160,200] 也可能是三个数：出处按两种读法都算
    assert _unsupported("有 160 条，另有 200 条。", '{"sizes": [16,160,200]}') == []
    assert _unsupported("一共 16,160,200 条。", '{"total": 16160200}') == []


def test_verdict_counts_every_mention():
    verdict = verify("成功 9 次，共 20 回合，成功率 0.61。", [BOARD])
    assert (len(verdict.mentions), [m.text for m in verdict.unsupported], verdict.ok) == (
        3,
        ["20", "0.61"],
        False,
    )
    assert verify("没有数字的回答。", []).ok and verify("没有数字的回答。", []).mentions == ()
    # 同一个数字写两次算两次：幻觉率（按数字）的分母是出现的次数
    assert len(verify("0.61 和 0.61", [BOARD]).unsupported) == 2
