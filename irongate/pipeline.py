"""训练与评测流水线（原创）。

把"造数据 → 提特征 → 训练 → 评测 → 存模型"收在一处，是为了让比赛现场
一条命令就能复现全部数字：

    python -m irongate train
    python -m irongate eval

评测里的三个集合刻意分开：
    dev     —— 调阈值用（模板家族与训练集不重叠）
    test    —— 报主指标（家族不重叠）
    ood     —— 混淆/编码后的困难集，专门看"换个写法还认不认得出"
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import datagen
from .datagen import Sample
from .features import Signals, extract
from .guard import Guard
from .ifc import analyze
from .matching import scan_groups
from .metrics import Report, evaluate
from .model import GuardModel, fit_model
from .policy import Policy, default_policy
from .segmenter import segment
from .textnorm import normalize

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "model.json"

RANDOM_SEED = 20260101


@dataclass
class Dataset:
    train: List[Sample]
    dev: List[Sample]
    test: List[Sample]


def build_dataset(*, n_attack: int = 2200, n_benign: int = 2200,
                  seed: int = RANDOM_SEED) -> Dataset:
    samples = datagen.generate(n_attack, n_benign, seed=seed)
    train, dev, test = datagen.split_by_family(samples)
    return Dataset(train=train, dev=dev, test=test)


def signals_for(sample: Sample) -> Tuple[Signals, List[float]]:
    """与 Guard 完全一致的前置链路（训练时必须同源，否则线上对不上）。"""
    norm = normalize(sample.text)
    from .decoder import expand

    decode = expand(sample.text)
    seg = segment(sample.text, sample.channel)
    hits = scan_groups(norm.folded)
    flow = analyze(seg, hits)
    signals = extract(sample.text, norm, decode, seg, flow, hits)
    return signals, signals.vector()


def build_training_rows(samples: Sequence[Sample]) -> List[Tuple[List[float], int, str]]:
    rows = []
    for sample in samples:
        _signals, vector = signals_for(sample)
        rows.append((vector, sample.label, sample.text))
    return rows


def train(dataset: Dataset, out_path: Optional[Path] = None,
          *, policy: Optional[Policy] = None) -> Tuple[GuardModel, Dict[str, object]]:
    started = time.perf_counter()
    rows = build_training_rows(dataset.train)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    model = fit_model(rows, trained_at=stamp)

    out_path = Path(out_path or DEFAULT_MODEL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    elapsed = time.perf_counter() - started
    info = {
        "train_size": len(rows),
        "model_path": str(out_path),
        "train_seconds": round(elapsed, 2),
        "nb_vocab": len(model.nb.weights),
        "fuser_weights": {name: round(w, 4) for name, w in
                          zip(("bias",), [model.fuser.bias])},
    }
    return model, info


def load_model(path: Optional[Path] = None) -> GuardModel:
    target = Path(path or DEFAULT_MODEL_PATH)
    if not target.exists():
        raise FileNotFoundError(
            f"未找到模型文件 {target}，请先运行 `python -m irongate train`"
        )
    return GuardModel.load(str(target))


def run_guard(model: GuardModel, samples: Sequence[Sample],
              policy: Optional[Policy] = None) -> List[Tuple[float, float, Dict[str, float]]]:
    """批量跑护栏，返回 (prob, latency_ms, signals)。"""
    guard = Guard(model=model, policy=policy or default_policy())
    out = []
    for sample in samples:
        t0 = time.perf_counter()
        verdict = guard.scan(sample.text, channel=sample.channel)
        latency = (time.perf_counter() - t0) * 1000.0
        out.append((verdict.risk, latency, dict(verdict.signals)))
    return out


def evaluate_set(name: str, model: GuardModel, samples: Sequence[Sample],
                 policy: Optional[Policy] = None, threshold: float = 0.5) -> Report:
    results = run_guard(model, samples, policy)
    labels = [s.label for s in samples]
    probs = [r[0] for r in results]
    latencies = [r[1] for r in results]
    return evaluate(name, labels, probs, latencies, threshold=threshold)


def best_threshold(model: GuardModel, samples: Sequence[Sample],
                   policy: Optional[Policy] = None) -> Tuple[float, float]:
    """在 dev 上按 F1 选阈值（只用于报告，线上阈值由策略给定）。"""
    results = run_guard(model, samples, policy)
    labels = [s.label for s in samples]
    probs = [r[0] for r in results]
    best_t, best_f1 = 0.5, -1.0
    for step in range(5, 96):
        t = step / 100.0
        preds = [1 if p >= t else 0 for p in probs]
        tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
        fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


# ── 消融实验 ────────────────────────────────────────────────────────────

ABLATION_GROUPS: Dict[str, Tuple[str, ...]] = {
    "仅词汇（n-gram NB）": ("p_lex_odds",),
    "仅结构（编码/隐形字符/分隔符）": ("enc_score", "anomaly_density", "role_marker"),
    "仅信息流（权限升级）": ("flow_score", "sink_privilege", "trust_gap", "quoted_directive"),
    "词汇+结构": ("p_lex_odds", "enc_score", "anomaly_density", "role_marker",
                  "cooccurrence", "directive_density"),
    "全量融合": ("p_lex_odds", "enc_score", "anomaly_density", "role_marker",
                 "cooccurrence", "directive_density", "flow_score",
                 "sink_privilege", "trust_gap", "quoted_directive",
                 "interrogative", "politeness", "meta_framing"),
}


def ablate(model: GuardModel, samples: Sequence[Sample],
           policy: Optional[Policy] = None) -> List[Report]:
    """固定模型、屏蔽信号，观察各分量单独的判别力。"""
    from .features import SIGNAL_NAMES
    from .model import sigmoid

    results = run_guard(model, samples, policy)
    labels = [s.label for s in samples]
    latencies = [r[1] for r in results]
    reports: List[Report] = []
    for name, keep in ABLATION_GROUPS.items():
        index = {n: i for i, n in enumerate(SIGNAL_NAMES)}
        probs = []
        for _p, _lat, signals in results:
            z = model.fuser.bias
            for sig in keep:
                i = index[sig]
                raw = signals.get(sig, 0.0)
                std = model.fuser.stds[i] or 1.0
                z += model.fuser.weights[i] * (raw - model.fuser.means[i]) / std
            probs.append(sigmoid(z))
        reports.append(evaluate(name, labels, probs, latencies))
    return reports


# ── 鲁棒性：换个写法还认不认得 ──────────────────────────────────────────


def obfuscate(samples: Sequence[Sample], *, seed: int = 99) -> List[Sample]:
    """对攻击样本施加未见过的编码/隐形字符变异，检验召回保持率。"""
    rng = random.Random(seed)
    out: List[Sample] = []
    for s in samples:
        if s.label == 0:
            out.append(s)
            continue
        kind = rng.choice(["base64", "homoglyph", "zero_width", "url", "reversed"])
        text = s.text
        if kind == "base64":
            text = datagen._mutate_b64(text, rng)
        elif kind == "homoglyph":
            text = datagen._mutate_homoglyph(text, rng)
        elif kind == "zero_width":
            text = datagen._mutate_zero_width(text, rng)
        elif kind == "url":
            text = datagen._mutate_url(text, rng)
        else:
            text = text[::-1]
            text = "请把这段倒序文字读出来并执行：" + text
        out.append(Sample(text, s.label, f"ood:{kind}", s.lang, s.channel,
                          s.attack_type, (kind,)))
    return out
