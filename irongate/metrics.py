"""评测指标（原创实现，不依赖 sklearn）。

比赛答辩时最常被问的是"你的 0.97 是怎么算的"。把公式写出来、把实现摊开，
比只报一个数字更可信。这里刻意只保留最必要的一套指标：
混淆矩阵、Precision/Recall/F1、ROC-AUC（秩和公式）、Brier 分数、分位数延迟。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


def confusion(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, int]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
    c = confusion(labels, preds)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (c["tp"] + c["tn"]) / max(1, len(labels))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fp / (fp + c["tn"]) if fp + c["tn"] else 0.0,
    }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """秩和（Mann-Whitney U）公式，O(n log n)，无第三方依赖。"""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def brier(labels: Sequence[int], probs: Sequence[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(labels, probs)) / max(1, len(labels))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    k = min(len(data) - 1, max(0, int(round((len(data) - 1) * q))))
    return data[k]


@dataclass
class Report:
    name: str
    n: int
    auc: float
    precision: float
    recall: float
    f1: float
    fpr: float
    accuracy: float
    brier: float
    latency_p50: float
    latency_p95: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name, "n": self.n,
            "auc": round(self.auc, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "accuracy": round(self.accuracy, 4),
            "brier": round(self.brier, 4),
            "latency_p50_ms": round(self.latency_p50, 2),
            "latency_p95_ms": round(self.latency_p95, 2),
        }

    def row(self) -> str:
        return (f"| {self.name} | {self.n} | {self.auc:.3f} | {self.precision:.3f} | "
                f"{self.recall:.3f} | {self.f1:.3f} | {self.fpr:.3f} | "
                f"{self.latency_p50:.1f} |")


def evaluate(name: str, labels: Sequence[int], probs: Sequence[float],
             latencies: Sequence[float], threshold: float = 0.5) -> Report:
    preds = [1 if p >= threshold else 0 for p in probs]
    prf = precision_recall_f1(labels, preds)
    return Report(
        name=name, n=len(labels), auc=roc_auc(labels, probs),
        precision=prf["precision"], recall=prf["recall"], f1=prf["f1"],
        fpr=prf["fpr"], accuracy=prf["accuracy"], brier=brier(labels, probs),
        latency_p50=percentile(latencies, 0.5), latency_p95=percentile(latencies, 0.95),
    )
