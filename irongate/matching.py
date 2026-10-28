"""词条匹配（原创，纯 str.find，无正则回溯风险）。

为什么不直接用 `re`：
词条里包含 `http://`、`rm -rf`、`eval(` 等含正则元字符的串，
逐个转义既啰嗦又容易出错；用 `str.find` 循环匹配既安全又快
（词条长度都很短，最坏情况退化为 O(n·m)，n 为文本长度）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .lexicons import GROUPS, ROLE_MARKERS


@dataclass(frozen=True)
class Hit:
    group: str
    term: str
    weight: float
    start: int
    end: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "group": self.group,
            "term": self.term,
            "weight": self.weight,
            "start": self.start,
            "end": self.end,
        }


def find_terms(text: str, terms: Sequence[Tuple[str, float]], group: str) -> List[Hit]:
    """不重叠地找出所有词条命中。"""
    hits: List[Hit] = []
    occupied = bytearray(len(text))
    for term, weight in sorted(terms, key=lambda t: len(t[0]), reverse=True):
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            end = idx + len(term)
            if not any(occupied[idx:end]):
                hits.append(Hit(group, term, weight, idx, end))
                for i in range(idx, end):
                    occupied[i] = 1
            start = idx + 1
    hits.sort(key=lambda h: h.start)
    return hits


def scan_groups(text: str, groups: Iterable[str] | None = None) -> List[Hit]:
    """扫描一个或多个词库分组。"""
    names = list(groups) if groups is not None else list(GROUPS)
    out: List[Hit] = []
    for name in names:
        out.extend(find_terms(text, GROUPS[name], name))
    out.sort(key=lambda h: (h.start, h.end))
    return out


def scan_role_markers(text: str) -> List[Hit]:
    return find_terms(text, ROLE_MARKERS, "role_marker")


def remap(hits: Sequence[Hit], index_map: Sequence[int]) -> List[Hit]:
    """把"紧凑视图"上的命中区间映射回原始文本坐标。"""
    out: List[Hit] = []
    for h in hits:
        if h.start >= len(index_map) or h.end - 1 >= len(index_map):
            continue
        start = index_map[h.start]
        end = index_map[h.end - 1] + 1
        out.append(Hit(h.group, h.term, h.weight, start, max(start + 1, end)))
    return out


def group_scores(hits: Sequence[Hit]) -> Dict[str, float]:
    """每个分组的最高权重（而不是累加），避免长文本堆叠造成虚高。"""
    scores: Dict[str, float] = {name: 0.0 for name in GROUPS}
    for hit in hits:
        scores[hit.group] = max(scores[hit.group], hit.weight)
    return scores


def group_counts(hits: Sequence[Hit]) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in GROUPS}
    for hit in hits:
        counts[hit.group] += 1
    return counts


def instruction_pressure(hits: Sequence[Hit]) -> float:
    """把命中压缩成一个 0~1 的"指令压力"：分组最高权重按 noisy-or 融合。"""
    scores = group_scores(hits)
    keep = ("override", "role", "authority", "exfil", "capability", "secrecy", "coercion")
    pressure = 0.0
    for name in keep:
        s = scores.get(name, 0.0)
        pressure = 1.0 - (1.0 - pressure) * (1.0 - 0.55 * s)
    return min(1.0, pressure)


def cooccurrence_bonus(folded: str) -> Tuple[float, List[Tuple[str, str]]]:
    """成对语义组合加成（"忽略 + 系统提示"这类组合比单词更能说明意图）。"""
    from .lexicons import CO_OCCURRENCE_BONUS

    total = 0.0
    matched: List[Tuple[str, str]] = []
    for a, b, bonus in CO_OCCURRENCE_BONUS:
        if a in folded and b in folded:
            total += bonus
            matched.append((a, b))
    return min(0.6, total), matched
