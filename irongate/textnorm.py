"""文本规范化与"形近/隐形字符"取证（原创实现）。

职责
----
把原始输入变成一份**等长折叠文本**，同时保留每一个异常字符的原始偏移，
使得后续的证据高亮可以直接映射回用户看到的原字符串。

为什么必须等长
--------------
常见做法是先 `NFKC` 再匹配，但 NFKC 会改变字符串长度（连字、组合字符），
导致匹配位置无法回指原文，证据高亮就会整体错位。这里改用**逐字符映射表**
（不可见字符→空格、形近字→ASCII、全角→半角、大写→小写），
只做一对一替换，因此 `len(folded) == len(raw)` 恒成立。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .lexicons import BIDI_CONTROLS, HOMOGLYPH_MAP, TAG_CHARS, ZERO_WIDTH

# 全角 ASCII 区间
_FULLWIDTH_LO, _FULLWIDTH_HI = 0xFF01, 0xFF5E
_FULLWIDTH_SPACE = 0x3000


@dataclass
class CharAnomaly:
    """一个异常字符。"""

    index: int
    kind: str  # zero_width | bidi | tag | homoglyph | fullwidth
    codepoint: int
    name: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "kind": self.kind,
            "codepoint": f"U+{self.codepoint:04X}",
            "name": self.name,
        }


@dataclass
class Normalization:
    """规范化结果。`folded` 与 `raw` 等长。"""

    raw: str
    folded: str
    anomalies: List[CharAnomaly] = field(default_factory=list)

    # 便捷统计
    @property
    def length(self) -> int:
        return len(self.raw)

    def count(self, kind: str) -> int:
        return sum(1 for a in self.anomalies if a.kind == kind)

    @property
    def anomaly_ratio(self) -> float:
        if not self.raw:
            return 0.0
        return min(1.0, len(self.anomalies) / max(1, len(self.raw)) * 12.0)

    def flags(self) -> Dict[str, float]:
        """给特征工程用的 0/1 或比例型信号。"""
        n = max(1, len(self.raw))
        return {
            "zero_width": 1.0 if self.count("zero_width") else 0.0,
            "bidi": 1.0 if self.count("bidi") else 0.0,
            "tag_chars": 1.0 if self.count("tag") else 0.0,
            "homoglyph": min(1.0, self.count("homoglyph") / n * 20.0),
            "fullwidth": 1.0 if self.count("fullwidth") else 0.0,
            "anomaly_density": self.anomaly_ratio,
        }


_KIND_NAMES = {
    "zero_width": "ZERO_WIDTH",
    "bidi": "BIDI_CONTROL",
    "tag": "TAG_CHARACTER",
    "homoglyph": "CONFUSABLE",
    "fullwidth": "FULLWIDTH",
}


def _is_fullwidth_alnum(cp: int) -> bool:
    """只把全角的**字母/数字**当作可疑折叠对象。

    中文正文里的全角标点（，。：；！？）是正常的，若也记为异常，
    任何一句正常中文都会被扣上"形近字符"的帽子。
    """
    return (0xFF10 <= cp <= 0xFF19) or (0xFF21 <= cp <= 0xFF3A) or (0xFF41 <= cp <= 0xFF5A)


def _is_fullwidth_punct(cp: int) -> bool:
    """中文正文常用的全角标点（！？：；（）等）：照常折叠成 ASCII 便于匹配，但不记为异常。"""
    return _FULLWIDTH_LO <= cp <= _FULLWIDTH_HI and not _is_fullwidth_alnum(cp)


def _classify(index: int, ch: str) -> CharAnomaly | None:
    cp = ord(ch)
    if cp in ZERO_WIDTH:
        return CharAnomaly(index, "zero_width", cp, _KIND_NAMES["zero_width"])
    if cp in BIDI_CONTROLS:
        return CharAnomaly(index, "bidi", cp, _KIND_NAMES["bidi"])
    if cp in TAG_CHARS:
        return CharAnomaly(index, "tag", cp, _KIND_NAMES["tag"])
    low = ch.lower()
    if _is_fullwidth_punct(cp):
        return None
    if low in HOMOGLYPH_MAP and HOMOGLYPH_MAP[low] != low:
        return CharAnomaly(index, "homoglyph", cp, _KIND_NAMES["homoglyph"])
    if _is_fullwidth_alnum(cp):
        return CharAnomaly(index, "fullwidth", cp, _KIND_NAMES["fullwidth"])
    return None


def normalize(text: str) -> Normalization:
    """折叠文本并记录异常字符。"""
    out: List[str] = []
    anomalies: List[CharAnomaly] = []
    for i, ch in enumerate(text):
        cp = ord(ch)
        anomaly = _classify(i, ch)
        if anomaly is not None:
            anomalies.append(anomaly)
        # 不可见类一律折叠为空格，避免"看起来是空白的恶意字符"拼接出新词
        if anomaly is not None and anomaly.kind in {"zero_width", "bidi", "tag"}:
            out.append(" ")
            continue
        if cp == _FULLWIDTH_SPACE:
            out.append(" ")
            continue
        if anomaly is not None and anomaly.kind == "fullwidth":
            out.append(chr(cp - 0xFEE0))
            continue
        low = ch.lower()
        if (anomaly is not None and anomaly.kind == "homoglyph") or (
                _is_fullwidth_punct(cp) and low in HOMOGLYPH_MAP):
            out.append(HOMOGLYPH_MAP[low])
            continue
        out.append(low if len(low) == 1 else ch)
    folded = "".join(out)
    # 兜底：极端情况下（字符大小写映射长度不为 1）仍强制等长
    if len(folded) != len(text):
        folded = folded[: len(text)].ljust(len(text))
    return Normalization(raw=text, folded=folded, anomalies=anomalies)


def script_profile(text: str) -> Dict[str, float]:
    """粗略统计字符集构成，用于识别"拉丁/西里尔混排"这类伪造迹象。"""
    latin = cjk = cyrillic = greek = other = 0
    for ch in text:
        cp = ord(ch)
        if 0x41 <= cp <= 0x7A:
            latin += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            cjk += 1
        elif 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif 0x0370 <= cp <= 0x03FF:
            greek += 1
        elif not ch.isspace():
            other += 1
    total = max(1, latin + cjk + cyrillic + greek + other)
    return {
        "latin_ratio": latin / total,
        "cjk_ratio": cjk / total,
        "cyrillic_ratio": cyrillic / total,
        "greek_ratio": greek / total,
        "other_ratio": other / total,
        "mixed_script": 1.0 if (cyrillic or greek) and latin else 0.0,
    }


def spans_of(anomalies: List[CharAnomaly], kind: str) -> List[Tuple[int, int]]:
    """把同类异常字符合并成连续区间（用于前端高亮）。"""
    spans: List[Tuple[int, int]] = []
    for a in anomalies:
        if a.kind != kind:
            continue
        if spans and spans[-1][1] == a.index:
            spans[-1] = (spans[-1][0], a.index + 1)
        else:
            spans.append((a.index, a.index + 1))
    return spans


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _is_ascii_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def compact_view(text: str) -> Tuple[str, List[int]]:
    """去掉"字符间插入的空格"，返回紧凑文本与**回指原串**的索引表。

    攻击者常用 `忽 略 以 上 指 令` 或 `i g n o r e` 这种"插空格"方式打断
    词条匹配。做法是在匹配用的紧凑视图上搜索，再把命中区间映射回原文，
    因此证据高亮依然准确，也不需要为每种插空方式各写一条规则。
    """
    keep_chars: List[str] = []
    index_map: List[int] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch == " " and 0 < i < n - 1:
            prev, nxt = text[i - 1], text[i + 1]
            if _is_cjk(prev) and _is_cjk(nxt):
                continue
            if _is_ascii_letter(prev) and _is_ascii_letter(nxt):
                left_ok = i - 2 < 0 or text[i - 2] in " \t，。：:,.!?！？；;"
                right_ok = i + 2 >= n or text[i + 2] in " \t，。：:,.!?！？；;"
                if left_ok and right_ok:
                    continue
        keep_chars.append(ch)
        index_map.append(i)
    return "".join(keep_chars), index_map
