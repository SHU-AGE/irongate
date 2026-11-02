"""信号提取：把一次分析压成 12 个**可读、可解释**的标量（原创）。

工业界的护栏常常把几十上百个特征塞进黑盒分类器，出了误判没人说得清原因。
这里刻意反其道而行：只保留 12 个语义明确的信号，每个都对应一句人话解释。
最终风险由一个小型线性融合器打分，于是"为什么判它恶意"可以直接
从权重 × 取值算出来，而不是靠 SHAP 事后解释。

信号分三类：
  * 词汇证据  p_lex（字符 n-gram 朴素贝叶斯的对数几率）
  * 结构异常  enc_score / anomaly_density / role_marker_max
  * 信息流    flow_score / sink_priv / trust_gap / quoted_directive
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .decoder import DecodeResult
from .ifc import FlowReport
from .matching import Hit, cooccurrence_bonus, group_scores
from .segmenter import Segmentation
from .textnorm import Normalization, script_profile

SIGNAL_NAMES: List[str] = [
    "bias",
    "p_lex_odds",
    "flow_score",
    "enc_score",
    "anomaly_density",
    "role_marker",
    "cooccurrence",
    "directive_density",
    "quoted_directive",
    "sink_privilege",
    "trust_gap",
    "interrogative",
    "politeness",
    "meta_framing",
]

SIGNAL_LABELS: Dict[str, str] = {
    "bias": "先验偏置",
    "p_lex_odds": "词汇证据（字符 n-gram 对数几率）",
    "flow_score": "权限流越界分",
    "enc_score": "分层编码风险",
    "anomaly_density": "隐形/形近字符密度",
    "role_marker": "伪造角色分隔符",
    "cooccurrence": "高危语义组合",
    "directive_density": "指令句密度",
    "quoted_directive": "引用内指令",
    "sink_privilege": "触及权限面等级",
    "trust_gap": "来源可信度缺口",
    "interrogative": "疑问句占比",
    "politeness": "礼貌外壳强度",
    "meta_framing": "元话语框定（分析/研究语境）",
}

# 元话语：把"引用攻击样本做分析"与"真的在攻击"区分开的唯一线索
META_DISCOURSE = (
    "分析", "解释", "研究", "论文", "防御", "不要执行", "别执行", "不要照做",
    "讲解", "科普", "复盘", "样例", "样本", "手法", "危害",
    "小说", "故事", "情节", "剧本", "设定里",
    "explain", "analyze", "analyse", "research", "paper", "defense", "defence",
    "do not follow", "don't follow", "do not run", "don't run", "case study",
    "how does", "why does", "detect", "guardrail",
    "story", "novel", "fiction", "screenplay", "in the story",
)

# 拒绝/禁止类措辞：出现在被引用指令附近时，说明这段文字是在**谈论**而不是**下达**
REFUSAL = (
    "拒绝", "不得", "不允许", "禁止", "不会执行", "不能执行",
    "refuse", "decline", "won't", "will not", "should not", "must not",
    "not allowed", "never",
)

# 只是语气/框定、不构成实际指令的言语行为
_SOFT_ACTS = frozenset({"polite", "framing"})


def _is_question(segments, index: int) -> bool:
    if not 0 <= index < len(segments):
        return False
    return any(ch in segments[index].text for ch in "？?")


@dataclass
class Signals:
    values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.values)

    def vector(self) -> List[float]:
        return [self.values.get(name, 0.0) for name in SIGNAL_NAMES]


def extract(
    text: str,
    norm: Normalization,
    decode: DecodeResult,
    seg: Segmentation,
    flow: FlowReport,
    hits: Sequence[Hit],
) -> Signals:
    """从四层分析结果里抽取 12 维信号。"""
    groups = group_scores(hits)
    profile = script_profile(norm.folded)
    bonus, _ = cooccurrence_bonus(norm.folded)

    role_marker = max(
        [h.weight for h in seg.role_markers] or [0.0]
    )
    directive_count = len(flow.directives)
    quoted_directive = 1.0 if any(d.quoted for d in flow.directives) else 0.0
    trust_gap = max([1.0 - d.trust for d in flow.directives] or [0.0])
    sink_priv = min(3, flow.max_privilege) / 3.0

    question_marks = sum(norm.raw.count(ch) for ch in "？?")
    interrogative = min(1.0, question_marks / max(1, len(seg.segments)))
    # "先问个无关问题？再下指令"：指令不在问句里时，问号不能替它开脱
    if any(d.act not in _SOFT_ACTS and not _is_question(seg.segments, d.segment_index)
           for d in flow.directives):
        interrogative = 0.0

    # 标点极少 + 指令词多，是"机器指令串"的典型形状
    density = min(1.0, directive_count / 3.0)

    polite_hits = sum(1 for h in hits if h.group == "polite")
    politeness = min(1.0, groups.get("polite", 0.0) + 0.08 * polite_hits)

    meta_ctx = quoted_directive > 0 or any(w in norm.folded for w in REFUSAL)
    meta = 1.0 if meta_ctx and any(w in norm.folded for w in META_DISCOURSE) else 0.0

    values = {
        "bias": 1.0,
        "p_lex_odds": 0.0,  # 由模型层回填（需要 NB 表）
        "flow_score": flow.score,
        "enc_score": decode.score(),
        "anomaly_density": min(1.0, norm.anomaly_ratio + profile["mixed_script"] * 0.6),
        "role_marker": role_marker,
        "cooccurrence": min(1.0, bonus),
        "directive_density": density,
        "quoted_directive": quoted_directive,
        "sink_privilege": sink_priv,
        "trust_gap": trust_gap,
        "interrogative": interrogative,
        "politeness": politeness,
        "meta_framing": meta,
    }
    return Signals(values=values)
