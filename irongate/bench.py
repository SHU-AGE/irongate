"""基准评测（原创）：交叉验证 + 消融 + 鲁棒性 + 外部挑战集。

为什么不用单次划分
------------------
单次划分的测试分数会随"哪几个模板家族被留出"大幅波动（本项目实测：
同一模型在不同划分上 F1 可以差 5 个点）。所以主指标一律用
**按模板家族分组的 5 折交叉验证**，报告均值 ± 标准差。

四个评测集
----------
1. `cv`        主指标：5 折分组交叉验证
2. `ablation`  同一模型逐组屏蔽信号，看各分量的独立贡献
3. `ood`       对攻击样本施加编码/隐形字符变异，看"换个写法"是否还认得出
4. `challenge` 人工撰写的困难样本（含大量"看着像攻击"的正常输入）
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .datagen import Sample, generate
from .guard import Guard
from .metrics import Report, evaluate
from .model import fit_model
from .pipeline import (ABLATION_GROUPS, build_dataset, build_training_rows,
                       obfuscate, run_guard)
from .policy import Policy, default_policy

CHALLENGE_PATH = Path(__file__).resolve().parent.parent / "bench" / "challenges.jsonl"


def _rows(samples: Sequence[Sample]) -> List[Tuple[List[float], int, str]]:
    return build_training_rows(samples)


def _score(model, samples: Sequence[Sample], policy: Policy) -> Tuple[List[float], List[float]]:
    results = run_guard(model, samples, policy)
    return [r[0] for r in results], [r[1] for r in results]


def grouped_folds(samples: Sequence[Sample], k: int, seed: int) -> List[Tuple[List[Sample], List[Sample]]]:
    """按模板家族（uid）分组切分，保证同一家族的样本不跨折。"""
    rng = random.Random(seed)
    uids = sorted({s.family for s in samples})
    rng.shuffle(uids)
    buckets: List[List[str]] = [[] for _ in range(k)]
    for i, uid in enumerate(uids):
        buckets[i % k].append(uid)
    folds = []
    for i in range(k):
        holdout = set(buckets[i])
        test = [s for s in samples if s.family in holdout]
        train = [s for s in samples if s.family not in holdout]
        if test and train:
            folds.append((train, test))
    return folds


def cross_validate(samples: Sequence[Sample], *, k: int = 5, seed: int = 20260101,
                   policy: Optional[Policy] = None) -> Dict[str, object]:
    policy = policy or default_policy()
    folds = grouped_folds(samples, k, seed)
    reports: List[Report] = []
    for idx, (train_rows, test_rows) in enumerate(folds, start=1):
        model = fit_model(_rows(train_rows))
        probs, lats = _score(model, test_rows, policy)
        labels = [s.label for s in test_rows]
        reports.append(evaluate(f"fold{idx}", labels, probs, lats))

    def agg(attr: str) -> Tuple[float, float]:
        values = [getattr(r, attr) for r in reports]
        return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)

    summary = {}
    for attr in ("auc", "precision", "recall", "f1", "fpr", "accuracy", "brier"):
        mean, std = agg(attr)
        summary[attr] = {"mean": round(mean, 4), "std": round(std, 4)}
    summary["folds"] = len(reports)
    summary["n"] = sum(r.n for r in reports)
    return summary


def ablation(samples: Sequence[Sample], model, policy: Optional[Policy] = None) -> List[Dict[str, object]]:
    from .features import SIGNAL_NAMES
    from .model import sigmoid

    policy = policy or default_policy()
    results = run_guard(model, samples, policy)
    labels = [s.label for s in samples]
    latencies = [r[1] for r in results]
    index = {n: i for i, n in enumerate(SIGNAL_NAMES)}
    out = []
    for name, keep in ABLATION_GROUPS.items():
        probs = []
        for _p, _lat, signals in results:
            z = model.fuser.bias
            for sig in keep:
                i = index[sig]
                std = model.fuser.stds[i] or 1.0
                z += model.fuser.weights[i] * (signals.get(sig, 0.0) - model.fuser.means[i]) / std
            probs.append(sigmoid(z))
        report = evaluate(name, labels, probs, latencies)
        out.append(report.as_dict())
    return out


def load_challenges(path: Path = CHALLENGE_PATH) -> List[Sample]:
    if not path.exists():
        return []
    samples: List[Sample] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        samples.append(Sample(
            text=item["text"], label=int(item["label"]), family=f"challenge#{i}",
            lang=item.get("lang", "zh"), channel=item.get("channel", "user"),
            attack_type=item.get("kind", ""),
        ))
    return samples


def _table(rows: Sequence[Report]) -> str:
    head = ("| 评测集 | 样本数 | AUC | Precision | Recall | F1 | FPR | P50 延迟(ms) |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    return head + "\n".join(r.row() for r in rows) + "\n"


def run_benchmark(model=None, *, seed: int = 20260101, folds: int = 5,
                  policy: Optional[Policy] = None) -> Dict[str, object]:
    """完整跑一遍并返回结构化结果 + Markdown。"""
    from .pipeline import train as train_pipeline

    policy = policy or default_policy()
    dataset = build_dataset(seed=seed)
    if model is None:
        model, _info = train_pipeline(dataset)

    # 1) 主指标：分组交叉验证（在每个训练折上重新拟合，避免"用测试集调参"）
    corpus = generate(2200, 2200, seed=seed)
    cv = cross_validate(corpus, k=folds, seed=seed, policy=policy)

    # 2) 单次留出（与 CV 对照，供文档展示）
    test_probs, test_lats = _score(model, dataset.test, policy)
    test_report = evaluate("test（单次留出）", [s.label for s in dataset.test],
                           test_probs, test_lats)

    # 3) 消融
    ablation_rows = ablation(dataset.test, model, policy)

    # 4) 鲁棒性：攻击样本变异后召回
    ood = obfuscate(dataset.test)
    ood_probs, ood_lats = _score(model, ood, policy)
    ood_report = evaluate("ood（编码/隐形变异）", [s.label for s in ood],
                          ood_probs, ood_lats)

    # 5) 人工挑战集
    challenges = load_challenges()
    challenge_report = None
    if challenges:
        ch_probs, ch_lats = _score(model, challenges, policy)
        challenge_report = evaluate("challenge（人工困难集）",
                                    [s.label for s in challenges], ch_probs, ch_lats)

    rows = [test_report, ood_report] + ([challenge_report] if challenge_report else [])
    markdown = ["## 基准结果\n", "### 单次留出 / 鲁棒性 / 挑战集\n",
                _table(rows),
                "### 分组 5 折交叉验证（主指标）\n",
                f"- AUC  {cv['auc']['mean']:.3f} ± {cv['auc']['std']:.3f}",
                f"- F1   {cv['f1']['mean']:.3f} ± {cv['f1']['std']:.3f}",
                f"- FPR  {cv['fpr']['mean']:.3f} ± {cv['fpr']['std']:.3f}",
                "",
                "### 消融（同一模型，逐组屏蔽信号）\n",
                "| 配置 | AUC | F1 | FPR | P50 延迟(ms) |",
                "| --- | ---: | ---: | ---: | ---: |"]
    for row in ablation_rows:
        markdown.append(f"| {row['name']} | {row['auc']:.3f} | {row['f1']:.3f} | "
                        f"{row['fpr']:.3f} | {row['latency_p50_ms']:.1f} |")

    return {
        "cv": cv,
        "test": test_report.as_dict(),
        "ood": ood_report.as_dict(),
        "challenge": challenge_report.as_dict() if challenge_report else None,
        "ablation": ablation_rows,
        "markdown": "\n".join(markdown),
    }
