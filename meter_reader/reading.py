"""十进制位权合并：重复位只核对，不相加；候选值与确认值分开。"""

from decimal import Decimal
import re

POINTER_PLACES = {
    "wheel3": [-1, -2, -3],
    "wheel4": [-1, -2, -3, -4],
    "pointer8d2": list(range(5, -3, -1)),
    "pointer8d3": list(range(4, -4, -1)),
    "pointer8d4": list(range(3, -5, -1)),
}


def pointer_sequence_issues(regions, layout, threshold):
    """Check every detected region before allowing carry-context decoding.

    These are consistency checks, not proof that local layout predictions are
    correct. Failed crops must stay in regions so missing dials cannot disappear.
    """
    if layout not in POINTER_PLACES:
        return ["表型或倍率未确认，未进行指针上下文解码"]
    issues = []
    pointers = [r for r in regions if r["kind"] == "pointer"]
    wheels = [r for r in regions if r["kind"] == "wheel"]
    expected = POINTER_PLACES[layout]
    if len(wheels) != int(layout.startswith("wheel")):
        issues.append("检测到的字轮数量与预测表型不一致")
    if len(pointers) != len(expected):
        issues.append(f"表型应有 {len(expected)} 个读数盘，检测到 {len(pointers)} 个")
    if any("quad" not in r or r.get("reason") for r in regions):
        issues.append("存在校正失败区域，未形成完整指针序列")
    if any(
        r.get("layout") != layout
        or r.get("weight_score", 0) < threshold
        or r.get("layout_score", 0) < threshold
        for r in regions
    ):
        issues.append("倍率或表型预测不一致或置信度不足")
    places = [r.get("exponent") for r in pointers]
    if len(set(places)) != len(places):
        issues.append("多个指针被预测为同一位权，无法确定指针顺序")
    if set(places) != set(expected):
        issues.append("指针位权缺失或超出表型范围，不能进行上下文进位")
    if any(r.get("exponent") != 0 for r in wheels):
        issues.append("字轮最低位倍率未确认为 10^0")
    return issues


def merge_places(places, highest, lowest, integer_width=1):
    values = {}
    reasons = []
    for place in places:
        e = place["exponent"]
        digit = str(place["digit"])
        if not re.fullmatch("[0-9]", digit):
            reasons.append(f"10^{e} 非数字")
            continue
        if e in values and values[e] != digit:
            reasons.append(f"10^{e} 重复位权冲突: {values[e]} / {digit}")
        else:
            values[e] = digit
    missing = [e for e in range(lowest, highest + 1) if e not in values]
    if missing:
        reasons.append("缺少位权: " + ", ".join(f"10^{e}" for e in missing))
    if any(e < lowest or e > highest for e in values):
        reasons.append("出现范围外位权")
    if reasons:
        return None, reasons
    number = sum((Decimal(v) * Decimal(10) ** e for e, v in values.items()), Decimal(0))
    text = format(number, f".{max(0,-lowest)}f")
    integer, sep, fraction = text.partition(".")
    return integer.zfill(integer_width) + (sep + fraction if sep else ""), []


def calculate(wheels, pointers, layout, reasons=None):
    reasons = list(reasons or [])
    notes = []
    places = []
    low = None
    result = dict(
        reading=None,
        candidate_reading=None,
        reading_candidates=[],
        unit="m³",
        status="needs_review",
        reasons=reasons,
        notes=notes,
    )
    if not layout:
        reasons.append("表型或倍率未确认")
        return result
    if layout.startswith("wheel"):
        lowest = -int(layout[-1])
        expected_count = -lowest
        if len(wheels) != 1:
            reasons.append("该表型需要一个字轮窗口")
            return result
        text = wheels[0].get("visible_text", "")
        if "v" in text:
            reasons.append("字轮出现 v 状态，保留可见标签，尚不能可靠消除翻字歧义")
            return result
        if not re.fullmatch(r"[0-9]{3,8}", text):
            reasons.append("字轮字符缺位或含非数字")
            return result
        highest = len(text) - 1
        width = len(text)
        places += [
            dict(exponent=len(text) - i - 1, digit=c) for i, c in enumerate(text)
        ]
        # No silent +/-1 of the wheel. Near carry boundaries expose ambiguity.
        fraction = {
            p["exponent"]: p.get("digit")
            for p in pointers
            if p.get("digit") is not None
        }
        if all(e in fraction for e in range(lowest, 0)):
            low = sum(
                (
                    Decimal(str(fraction[e])) * Decimal(10) ** e
                    for e in range(lowest, 0)
                ),
                Decimal(0),
            )
            # Public training labels contain plain (non-v) windows already
            # showing the next integer while the pointer fraction is still high.
            # Without a validated transition resolver, retain this ambiguity.
            if low >= Decimal(".5") or low < Decimal(".02"):
                reasons.append("字轮可能提前翻位或处于9→0临界区，需核对整数与指针衔接")
                notes.append("没有自动把字轮加一或减一")
    elif layout in ("pointer8d2", "pointer8d3", "pointer8d4"):
        lowest = -int(layout[-1])
        highest = 7 + lowest
        expected_count = 8
        width = highest + 1
        if wheels:
            reasons.append("全指针表型中检测到字轮，表型不一致")
    else:
        reasons.append(f"不支持的表型: {layout}")
        return result
    if len(pointers) != expected_count:
        reasons.append(f"表型应有 {expected_count} 个读数盘，实际 {len(pointers)} 个")
    for p in pointers:
        if p.get("digit") is None:
            reasons.append("部分指针没有可用的上下文数字")
            continue
        places.append(dict(exponent=p["exponent"], digit=p["digit"]))
        if p.get("raw_class", "")[:1] != p["digit"]:
            notes.append(
                f"10^{p['exponent']}: 单盘 {p.get('raw_class')} → 作者上下文模型 {p['digit']}"
            )
    candidate, issues = merge_places(places, highest, lowest, width)
    reasons.extend(issues)
    result["candidate_reading"] = candidate
    result["reading_candidates"] = [candidate] if candidate is not None else []
    if (
        candidate is not None
        and layout.startswith("wheel")
        and low is not None
        and low >= Decimal(".5")
    ):
        previous_integer = int(text) - 1
        if previous_integer >= 0:
            alternative = (
                str(previous_integer).zfill(width) + "." + candidate.split(".")[1]
            )
            result["reading_candidates"].append(alternative)
    if (
        candidate is not None
        and layout.startswith("wheel")
        and low is not None
        and low < Decimal(".02")
    ):
        next_integer = int(text) + 1
        if next_integer < 10**width:
            result["reading_candidates"].append(
                str(next_integer).zfill(width) + "." + candidate.split(".")[1]
            )
    if candidate is not None and not reasons:
        result.update(reading=candidate, status="ok")
    result["reasons"] = list(dict.fromkeys(reasons))
    return result


def examples():
    examples = [
        (
            "重复位权不重复相加",
            [
                dict(exponent=0, digit="2"),
                dict(exponent=0, digit="2"),
                dict(exponent=-1, digit="3"),
            ],
            0,
            -1,
            "2.3",
        ),
        (
            "9→0 后的十进制组合",
            [
                dict(exponent=1, digit="1"),
                dict(exponent=0, digit="0"),
                dict(exponent=-1, digit="0"),
            ],
            1,
            -1,
            "10.0",
        ),
        ("缺位保留空值", [dict(exponent=0, digit="1")], 0, -1, None),
        (
            "重复位权冲突",
            [dict(exponent=0, digit="1"), dict(exponent=0, digit="2")],
            0,
            0,
            None,
        ),
    ]
    for name, places, hi, lo, expected in examples:
        value, reasons = merge_places(places, hi, lo)
        assert value == expected, (name, value)
        print(name, value, reasons)
    p = [
        dict(exponent=-1, digit="9", raw_class="9"),
        dict(exponent=-2, digit="9", raw_class="9"),
        dict(exponent=-3, digit="9", raw_class="9"),
    ]
    assert calculate([dict(visible_text="00123")], p, "wheel3")["reading"] is None
    assert calculate([dict(visible_text="0012v3")], p, "wheel3")["reading"] is None
    early = calculate(
        [dict(visible_text="00481")],
        [dict(exponent=-i - 1, digit=c, raw_class=c) for i, c in enumerate("8333")],
        "wheel4",
    )
    assert early["reading"] is None and early["reading_candidates"] == [
        "00481.8333",
        "00480.8333",
    ]
    print("提前翻字候选", early["reading_candidates"])
    print("临界进位和未知 v 状态均保留待核对")


if __name__ == "__main__":
    examples()
