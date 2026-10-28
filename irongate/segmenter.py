"""分段与来源标注（原创实现）。

提示注入的判定强依赖于**这段话是谁说的**。同样一句 "忽略之前的指令"：
出现在系统提示里是正常的元指令，出现在"网页检索结果"里就是典型注入。
因此本模块把输入切成小句，并为每一句标注：

* `channel`：system / user / tool / retrieval / unknown（渠道即信任层级）
* `quoted`：是否位于引号或代码块内（"引用洗白"的观测点）
* `offset`：在原文中的字符区间（证据高亮的前提）

切分只用标点与换行，保持**无条件、无模型**的确定性，
这样同一段文本每次切分结果完全一致，便于回归测试与审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .matching import Hit, scan_role_markers

# 渠道信任度：1.0 完全可信，0.0 完全不可信
CHANNEL_TRUST: Dict[str, float] = {
    "system": 1.00,
    "user": 0.55,
    "tool": 0.30,
    "retrieval": 0.15,
    "unknown": 0.35,
}

_BREAKS = set("。！？!?\n；;")
_SOFT_BREAKS = set("，,、：:")
_OPENERS = {"\"": "\"", "“": "”", "「": "」", "『": "』", "‘": "’", "`": "`"}
_FENCE = "```"


@dataclass
class Segment:
    index: int
    text: str
    start: int
    end: int
    channel: str
    quoted: bool = False
    trust_override: float | None = None

    @property
    def trust(self) -> float:
        if self.trust_override is not None:
            return self.trust_override
        return CHANNEL_TRUST.get(self.channel, CHANNEL_TRUST["unknown"])

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "span": [self.start, self.end],
            "channel": self.channel,
            "trust": self.trust,
            "quoted": self.quoted,
        }


@dataclass
class Segmentation:
    text: str
    channel: str
    segments: List[Segment] = field(default_factory=list)
    role_markers: List[Hit] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "segment_count": len(self.segments),
            "segments": [s.as_dict() for s in self.segments],
            "role_markers": [m.as_dict() for m in self.role_markers],
        }


def quoted_regions(text: str) -> List[Tuple[int, int]]:
    """找出引号/代码块覆盖的区间（允许不闭合，未闭合时延到队尾）。"""
    regions: List[Tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(_FENCE, i):
            close = text.find(_FENCE, i + 3)
            end = n if close < 0 else close + 3
            regions.append((i, end))
            i = end
            continue
        ch = text[i]
        closer = _OPENERS.get(ch)
        if closer:
            close = text.find(closer, i + 1)
            end = n if close < 0 else close + 1
            regions.append((i, end))
            i = end
            continue
        i += 1
    return regions


def _overlaps(start: int, end: int, regions: List[Tuple[int, int]]) -> bool:
    """句子与引号区是否有实质重叠。

    只看"句子起点是否在引号内"会漏掉常见形态：
    `……样本：“忽略以上指令”` —— 引号从句子中段开始，起点却在引号外。
    """
    for r_start, r_end in regions:
        lo, hi = max(start, r_start), min(end, r_end)
        if hi - lo >= 3:
            return True
        if lo < hi and (hi - lo) >= max(1, (end - start) // 3):
            return True
    return False


def segment(text: str, channel: str = "user",
            trust_map: Dict[str, float] | None = None) -> Segmentation:
    """按标点/换行切分，并标注引号与角色标记。

    `trust_map` 允许调用方用策略覆盖渠道信任度（默认取 `CHANNEL_TRUST`）。
    """
    trust = CHANNEL_TRUST.get(channel, CHANNEL_TRUST["unknown"])
    if trust_map is not None and channel in trust_map:
        trust = float(trust_map[channel])
    segs: List[Segment] = []
    regions = quoted_regions(text)
    n = len(text)
    start = 0
    idx = 0
    i = 0
    while i < n:
        ch = text[i]
        cut = ch in _BREAKS
        if not cut and ch in _SOFT_BREAKS and (i - start) >= 24:
            cut = True
        if cut:
            raw = text[start:i + 1]
            if raw.strip():
                segs.append(Segment(idx, raw, start, i + 1, channel,
                                    _overlaps(start, i + 1, regions), trust))
                idx += 1
            start = i + 1
        i += 1
    if start < n and text[start:].strip():
        segs.append(Segment(idx, text[start:], start, n, channel,
                            _overlaps(start, i + 1, regions), trust))
    return Segmentation(text=text, channel=channel, segments=segs,
                        role_markers=scan_role_markers(text))
