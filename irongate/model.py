"""轻量检测模型：字符 n-gram 朴素贝叶斯 + 线性信号融合（原创，零依赖）。

为什么不用神经网络
------------------
1. **可复现**：比赛评审最怕"跑不起来"。本实现只用标准库，训练 30 秒内完成；
2. **可解释**：最终分数 = Σ w_i · s_i，每一项都能翻译成一句人话；
3. **抗过拟合**：合成语料规模有限，深层模型只会把模板背下来。

两段式结构
----------
第一段：字符 n-gram（2/3-gram）伯努利朴素贝叶斯，给出"这段文字像不像注入"
        的对数几率 `p_lex_odds`；
第二段：把 `p_lex_odds` 与结构异常、信息流越界等 12 个信号一起送入
        一个 13 维逻辑回归（我们称之为 **融合器**），输出校准后的风险概率。

第一段负责"措辞"，第二段负责"结构 + 权限语义"。两者互补：
纯编码载荷的措辞是乱码（第一段低分），但编码层数与权限面会把它拉回来。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .features import SIGNAL_NAMES, Signals

# NB 对数几率的压缩尺度。尺度太小会让所有攻击都饱和到 +1，
# 融合器就再也分不出"很像攻击"和"铁定是攻击"，概率校准随之崩掉。
LEX_SCALE = 2.0

# ── 工具函数 ────────────────────────────────────────────────────────────


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _char_ngrams(text: str, sizes: Tuple[int, ...] = (2, 3), cap: int = 480) -> List[str]:
    t = "".join(ch for ch in text.lower() if ch.isprintable())
    grams: List[str] = []
    for n in sizes:
        if len(t) < n:
            continue
        for i in range(len(t) - n + 1):
            grams.append(t[i:i + n])
            if len(grams) >= cap:
                return grams
    return grams


# ── 第一段：朴素贝叶斯 ──────────────────────────────────────────────────


@dataclass
class NaiveBayes:
    """伯努利朴素贝叶斯（存在性特征），权重即对数几率。

    两个关键工程处理：

    * **长度归一**：只看"命中了多少词条"会让长文本天然得分更高（攻击样本
      往往更长，模型于是学会"长 = 恶意"）。这里用 `1/√n` 归一；
    * **输出标准化**：训练结束后用训练集上的均值/方差把原始对数几率变成
      z 分数，避免 `tanh` 在进入融合器之前就把绝大多数样本压到 ±1（信息全丢）。
    """

    prior: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    sizes: Tuple[int, ...] = (2, 3)
    mu: float = 0.0
    sigma: float = 1.0

    def _raw(self, text: str) -> float:
        grams = set(_char_ngrams(text, self.sizes))
        score = self.prior
        weights = self.weights
        for g in grams:
            w = weights.get(g)
            if w:
                score += w
        return score / math.sqrt(max(1, len(grams)))

    def train(self, docs: Sequence[Tuple[str, int]]) -> None:
        pos_docs = neg_docs = 0
        pos: Dict[str, int] = {}
        neg: Dict[str, int] = {}
        for text, label in docs:
            grams = set(_char_ngrams(text, self.sizes))
            if label:
                pos_docs += 1
                for g in grams:
                    pos[g] = pos.get(g, 0) + 1
            else:
                neg_docs += 1
                for g in grams:
                    neg[g] = neg.get(g, 0) + 1
        self.prior = math.log((pos_docs + 1) / (neg_docs + 1))

        vocab = set(pos) | set(neg)
        raw: Dict[str, float] = {}
        for g in vocab:
            p_pos = (pos.get(g, 0) + 1.0) / (pos_docs + 2.0)
            p_neg = (neg.get(g, 0) + 0.5) / (neg_docs + 1.0)
            raw[g] = math.log(p_pos / p_neg)
        # 剪枝：只保留"支持度 × 信息量"最高的词条，控制模型体积
        self.weights = {
            g: round(w, 4)
            for g, w in sorted(
                raw.items(),
                key=lambda kv: abs(kv[1]) * min(pos.get(kv[0], 0) + neg.get(kv[0], 0), 40),
                reverse=True,
            )[:20000]
            if abs(w) >= 0.12
        }

        scores = [self._raw(text) for text, _ in docs]
        n = max(1, len(scores))
        self.mu = sum(scores) / n
        var = sum((s - self.mu) ** 2 for s in scores) / n
        self.sigma = math.sqrt(var) or 1.0

    def log_odds(self, text: str) -> float:
        """返回标准化后的对数几率（约 N(0,1) 量级）。"""
        return (self._raw(text) - self.mu) / (self.sigma or 1.0)

    def save(self) -> Dict[str, object]:
        return {"prior": round(self.prior, 4), "sizes": list(self.sizes),
                "mu": round(self.mu, 4), "sigma": round(self.sigma, 4),
                "weights": self.weights}

    @classmethod
    def load(cls, data: Dict[str, object]) -> "NaiveBayes":
        return cls(prior=float(data["prior"]),
                   weights={k: float(v) for k, v in dict(data["weights"]).items()},
                   sizes=tuple(data.get("sizes", (2, 3))),
                   mu=float(data.get("mu", 0.0)),
                   sigma=float(data.get("sigma", 1.0)))


# ── 第二段：融合器 ──────────────────────────────────────────────────────


@dataclass
class Fuser:
    """对 12 个语义明确信号做**标准化后的**逻辑回归。

    为什么要标准化：各信号量纲差异巨大（`p_lex_odds` 是 tanh 压过的对数几率，
    `interrogative` 是 0~1 的占比）。不标准化时，权重量级完全由方差决定，
    训练会退化成"词汇证据一枝独秀"，结构信号与信息流信号被压成噪声。
    """

    weights: List[float] = field(default_factory=lambda: [0.0] * len(SIGNAL_NAMES))
    bias: float = 0.0
    means: List[float] = field(default_factory=lambda: [0.0] * len(SIGNAL_NAMES))
    stds: List[float] = field(default_factory=lambda: [1.0] * len(SIGNAL_NAMES))

    def _z(self, values: Sequence[float]) -> List[float]:
        return [(v - m) / (s if s > 1e-9 else 1.0)
                for v, m, s in zip(values, self.means, self.stds)]

    def standardize(self, signals: Signals) -> List[float]:
        raw = [signals.values.get(name, 0.0) for name in SIGNAL_NAMES]
        return self._z(raw)

    def score(self, signals: Signals) -> float:
        z = self.standardize(signals)
        return self.bias + sum(w * v for w, v in zip(self.weights, z))

    def prob(self, signals: Signals) -> float:
        return sigmoid(self.score(signals))

    def contributions(self, signals: Signals) -> List[Dict[str, object]]:
        """逐信号的贡献（权重 × 标准化取值），用于解释面板。"""
        out = []
        z = self.standardize(signals)
        for name, w, zi in zip(SIGNAL_NAMES, self.weights, z):
            out.append({
                "signal": name,
                "value": round(signals.values.get(name, 0.0), 4),
                "z": round(zi, 3),
                "weight": round(w, 4),
                "contribution": round(w * zi, 4),
            })
        out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return out

    def fit(self, rows: Sequence[Tuple[List[float], int]], *,
            epochs: int = 1500, lr: float = 0.18, l2: float = 8e-3,
            seed: int = 7) -> List[float]:
        """全批量梯度下降（参数只有 14 个，秒级收敛）。

        带**部分符号约束**：结构类与权限类信号（编码层数、隐形字符密度、
        伪造分隔符、引用内指令……）在设计上只表示"风险上升"，因此把它们的
        权重投影到非负象限——否则样本相关性会把"高危语义组合"学成负权重，
        解释面板出现"命中高危组合，贡献 -0.7"这种自相矛盾的输出。

        词汇证据、权限流总分与元话语允许自由取负：前者是校准的主力，
        后者本身就是"反证"（分析语境越长越像正常讨论）。
        """
        rng = random.Random(seed)
        data = list(rows)
        rng.shuffle(data)
        n = max(1, len(data))
        cols = list(zip(*[x for x, _ in data])) if data else [[] for _ in SIGNAL_NAMES]
        self.means = [sum(col) / n for col in cols]
        self.stds = []
        for col, mean in zip(cols, self.means):
            var = sum((v - mean) ** 2 for v in col) / n
            self.stds.append(math.sqrt(var) or 1.0)

        negative_ok = {SIGNAL_NAMES.index(n) for n in (
            "p_lex_odds", "flow_score", "interrogative", "politeness", "meta_framing",
        )}
        z_rows = [(self._z(x), y) for x, y in data]
        losses: List[float] = []
        for epoch in range(epochs):
            grad = [0.0] * len(self.weights)
            grad_b = 0.0
            loss = 0.0
            for x, y in z_rows:
                z = self.bias + sum(w * v for w, v in zip(self.weights, x))
                p = sigmoid(z)
                p = min(1 - 1e-9, max(1e-9, p))
                loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                d = (p - y) / n
                grad_b += d
                for i, v in enumerate(x):
                    if v:
                        grad[i] += d * v
            for i in range(len(self.weights)):
                grad[i] += l2 * self.weights[i]
                self.weights[i] -= lr * grad[i]
                if i not in negative_ok:
                    self.weights[i] = max(0.0, self.weights[i])
            self.bias -= lr * grad_b
            if epoch % 50 == 0:
                losses.append(loss / n)
        return losses

    def save(self) -> Dict[str, object]:
        return {
            "weights": [round(w, 5) for w in self.weights],
            "bias": round(self.bias, 5),
            "means": [round(m, 5) for m in self.means],
            "stds": [round(s, 5) for s in self.stds],
        }

    @classmethod
    def load(cls, data: Dict[str, object]) -> "Fuser":
        return cls(
            weights=[float(w) for w in data["weights"]],
            bias=float(data["bias"]),
            means=[float(m) for m in data.get("means", [0.0] * len(SIGNAL_NAMES))],
            stds=[float(s) for s in data.get("stds", [1.0] * len(SIGNAL_NAMES))],
        )


# ── 组合模型 ────────────────────────────────────────────────────────────


@dataclass
class GuardModel:
    nb: NaiveBayes = field(default_factory=NaiveBayes)
    fuser: Fuser = field(default_factory=Fuser)
    trained_at: str = ""
    train_size: int = 0
    version: int = 1

    def signals_with_lex(self, signals: Signals, folded_text: str) -> Signals:
        """把 NB 的对数几率写入 signals（用 tanh 压到 [-1,1]）。"""
        odds = self.nb.log_odds(folded_text)
        signals.values["p_lex_odds"] = math.tanh(odds / LEX_SCALE)
        return signals

    def predict(self, signals: Signals) -> float:
        return self.fuser.prob(signals)

    def save(self, path: str) -> None:
        payload = {
            "version": self.version,
            "trained_at": self.trained_at,
            "train_size": self.train_size,
            "nb": self.nb.save(),
            "fuser": self.fuser.save(),
            "signal_names": SIGNAL_NAMES,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "GuardModel":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(nb=NaiveBayes.load(data["nb"]),
                   fuser=Fuser.load(data["fuser"]),
                   trained_at=data.get("trained_at", ""),
                   train_size=int(data.get("train_size", 0)))


def fit_model(rows: Sequence[Tuple[List[float], int, str]],
              *, trained_at: str = "") -> GuardModel:
    """两段式训练。

    融合器如果直接吃"在全量训练集上训练的 NB 打分"，会把这个分数当作
    近乎完美的特征（信息泄漏），学到的权重会失真。这里采用**分层堆叠**：
    用一半样本训练一个辅助 NB，给另一半样本算 `p_lex_odds` 来训练融合器，
    最后再用全量样本重训 NB 作为线上模型。
    """
    rows = list(rows)
    half = len(rows) // 2
    a, b = rows[:half], rows[half:]

    aux = NaiveBayes()
    aux.train([(text, label) for _vec, label, text in a])

    lex_idx = SIGNAL_NAMES.index("p_lex_odds")
    stack_rows: List[Tuple[List[float], int]] = []
    for vec, label, text in b:
        v = list(vec)
        v[lex_idx] = math.tanh(aux.log_odds(text) / LEX_SCALE)
        stack_rows.append((v, label))

    fuser = Fuser()
    fuser.fit(stack_rows)

    full_nb = NaiveBayes()
    full_nb.train([(text, label) for _vec, label, text in rows])
    return GuardModel(nb=full_nb, fuser=fuser, trained_at=trained_at,
                      train_size=len(rows))

