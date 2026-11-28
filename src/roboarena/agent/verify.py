"""结果校验：回答里的每个数字，必须能在本次会话的工具输出或用户输入里找到出处（docs/PLAN.md 1.2 节）。

纯函数。规则（细节与例子在 tests/test_agent_verify.py）：
- 有出处：数字与出处里的某个数相等；或者是它四舍五入到回答里写出的位数，且至少保住它的一位有效数字
  （0.5625 可以说成 0.56、0.6，不能说成 1）；带百分号的数字可以是出处里某个数的 100 倍，
  出处里带百分号的数也可以写成小数。只比大小，不比正负。
- 不计：列表与标题的编号、「第 N 步 / 点 / 条」这类只起列举作用的序数；紧跟在英文字母后面的数字
  （名称的一部分，如 bc-step10000、qwen08b）。名次（「第 3 名」）要有出处。出处那一侧不做这些排除。
- 只认阿拉伯数字；中文数字（「三个」）不在检查范围内。只管数字，不管文字性的错误。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

_NUMBER = re.compile(
    r"(?P<int>\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.(?P<frac>\d+))?"
    r"(?:[eE](?P<exp>[+-]?\d+)(?![A-Za-z0-9]))?"
    r"(?P<percent>\s?[%％])?"
)
# 行首的编号：可以带标题的井号、列表的短横，如「1. 」「3) 」「（4）」「- 5、」「## 2. 」。「0.5」这样后面还有数字的不是编号
_ENUMERATOR = re.compile(r"^(\s*(?:#{1,6}\s*)?(?:[-*•]\s+)?)[（(]?\d{1,3}[)）.、](?!\d)", re.MULTILINE)
_ORDINAL = re.compile(r"第\s*\d+\s*(?:步|点|条|项|节|章|部分)")
_EPS = 1e-9


@dataclass(frozen=True)
class Mention:
    text: str  # 回答里的原文，如 "56.3%"
    value: float  # 数值（不带正负号；百分号不换算）
    unit: float  # 写出来的最后一位的大小，如 0.460 → 0.001
    percent: bool


@dataclass(frozen=True)
class Verdict:
    mentions: tuple[Mention, ...]  # 回答里携带事实的数字，重复出现的各算一次
    unsupported: tuple[Mention, ...]  # 其中找不到出处的

    @property
    def ok(self) -> bool:
        return not self.unsupported


def _in_name(text: str, start: int) -> bool:
    """这个数字是不是紧跟在英文字母（或下划线）后面：是的话它是名称的一部分，如 step20、qwen08b、noise0.3。"""
    return start > 0 and (text[start - 1].isascii() and (text[start - 1].isalpha() or text[start - 1] == "_"))


def _numbers(text: str, *, skip_names: bool = False) -> list[Mention]:
    found = []
    for m in _NUMBER.finditer(text):
        if skip_names and _in_name(text, m.start()):
            continue
        frac, exp = m.group("frac") or "", int(m.group("exp") or 0)
        digits = m.group("int").replace(",", "")
        value = float(f"{digits}.{frac or '0'}e{exp}")
        found.append(
            Mention(m.group(0).strip(), value, 10.0 ** (exp - len(frac)), m.group("percent") is not None)
        )
    return found


def mentions(answer: str) -> tuple[Mention, ...]:
    """回答里携带事实的数字。编号与列举用的序数不算。"""
    text = _ENUMERATOR.sub(lambda m: m.group(1), answer)
    return tuple(_numbers(_ORDINAL.sub("", text), skip_names=True))


def _source_values(sources: Sequence[str]) -> list[float]:
    values: list[float] = []
    for source in sources:
        for number in _numbers(source):
            values.append(number.value)
            if number.percent:
                values.append(number.value / 100)
            if "," in number.text:  # 紧凑的 JSON 里 [16,160,200] 也可能是三个数：两种读法都算
                values += [float(part) for part in re.findall(r"\d+", number.text.split(".")[0])]
    return values


def _matches(value: float, unit: float, source: float) -> bool:
    if math.isclose(value, source, rel_tol=1e-12, abs_tol=0.0) or value == source:
        return True
    if source == 0:
        return False
    leading = 10.0 ** math.floor(math.log10(source) + 1e-12)  # 出处那个数最高位的大小
    if unit > leading * (1 + _EPS):  # 四舍五入把它整个吞掉了
        return False
    return abs(value - source) <= unit / 2 * (1 + _EPS)


def _supported(mention: Mention, values: Sequence[float]) -> bool:
    if any(_matches(mention.value, mention.unit, v) for v in values):
        return True
    if mention.percent and any(_matches(mention.value / 100, mention.unit / 100, v) for v in values):
        return True
    if "," in mention.text:  # 「16,160」也可能是并列的两个数：每一段都有出处也算
        parts = re.findall(r"\d+", mention.text.split(".")[0])
        return all(any(_matches(float(part), 1.0, v) for v in values) for part in parts)
    return False


def says(answer: str, value: float) -> bool:
    """回答里有没有说到 value 这个数：相等、四舍五入后相等，或写成了百分比。AI 评测核对事实时用。"""
    target = abs(value)
    return any(
        _matches(m.value, m.unit, target) or (m.percent and _matches(m.value / 100, m.unit / 100, target))
        for m in mentions(answer)
    )


def verify(answer: str, sources: Sequence[str]) -> Verdict:
    """sources 是本次会话里全部的工具输出与用户输入（原文）。"""
    values = _source_values(sources)
    found = mentions(answer)
    return Verdict(found, tuple(m for m in found if not _supported(m, values)))
