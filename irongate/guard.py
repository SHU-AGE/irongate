"""护栏编排：把四层分析合成一次可解释的判决（原创）。

调用链
------
    normalize → decode(多层) → segment(来源标注) → match/IFC → model → policy

设计约定
--------
* **检测与动作分离**：本模块只产出 risk 与证据，动作由 `Policy` 决定；
* **解释优先**：每个判决都带 `contributions`（权重×取值）与 `evidence`
  （原文区间），前端可以逐条高亮；
* **失败安全**：任何内部分析异常都会退化为"隔离"而不是"放行"，
  因为"检测器崩了"不应该等价于"攻击成功"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import features
from .decoder import DecodeResult, expand
from .features import SIGNAL_LABELS, Signals
from .ifc import (ConversationTracker, FlowReport, OutputReport, analyze,
                  verify_output)
from .matching import Hit, remap, scan_groups
from .model import GuardModel
from .policy import Policy, default_policy
from .segmenter import Segmentation, segment
from .textnorm import Normalization, compact_view, normalize

_HIGH_RISK_ACTS = {"override", "authority", "exfil", "capability", "role"}


@dataclass
class Evidence:
    start: int
    end: int
    kind: str
    label: str
    detail: str = ""
    severity: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "span": [self.start, self.end],
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "severity": round(self.severity, 3),
        }


@dataclass
class Verdict:
    risk: float
    action: str
    channel: str
    model_prob: float
    flow_score: float
    signals: Dict[str, float]
    contributions: List[Dict[str, object]]
    evidence: List[Evidence]
    decode: Dict[str, object]
    flow: Dict[str, object]
    sanitized: str
    removed_segments: List[int]
    allowed_capabilities: List[str]
    explanation: str
    latency_ms: float
    truncated: bool = False
    error: Optional[str] = None
    decision_source: str = "rule"
    decision_source_label: str = "规则"
    deepseek: Optional[Dict[str, object]] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "risk": round(self.risk, 4),
            "action": self.action,
            "channel": self.channel,
            "model_prob": round(self.model_prob, 4),
            "flow_score": round(self.flow_score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "contributions": self.contributions,
            "evidence": [e.as_dict() for e in self.evidence],
            "decode": self.decode,
            "flow": self.flow,
            "sanitized": self.sanitized,
            "removed_segments": self.removed_segments,
            "allowed_capabilities": self.allowed_capabilities,
            "explanation": self.explanation,
            "latency_ms": round(self.latency_ms, 2),
            "truncated": self.truncated,
            "error": self.error,
            "decision_source": self.decision_source,
            "decision_source_label": self.decision_source_label,
            "deepseek": self.deepseek,
        }


@dataclass
class OutputVerdict:
    report: OutputReport
    action: str
    risk: float

    def as_dict(self) -> Dict[str, object]:
        return {"action": self.action, "risk": round(self.risk, 3), **self.report.as_dict()}


def _merge_flow(main: FlowReport, extra: FlowReport) -> FlowReport:
    """多次分析（原文 + 解码展开）合并：保留较高分的那一份证据。"""
    if extra.score <= main.score:
        return main
    return extra


def _evidence_from_hits(hits: Sequence[Hit]) -> List[Evidence]:
    out: List[Evidence] = []
    for h in hits:
        if h.weight < 0.55:
            continue
        out.append(Evidence(h.start, h.end, f"lexicon:{h.group}", h.term,
                            f"命中 {h.group} 词条（强度 {h.weight:.2f}）", h.weight))
    return out


def _explain(risk: float, action: str, contributions: List[Dict[str, object]],
             flow: FlowReport, decode: DecodeResult) -> str:
    """把判决翻译成人话（用于日志、控制台与比赛答辩）。"""
    action_text = {
        "allow": "放行",
        "sanitize": "净化后放行",
        "quarantine": "隔离（只读上下文，不触发工具）",
        "block": "拦截",
    }[action]
    parts = [f"判定动作：{action_text}；综合风险 {risk:.2f}。"]
    positive = [c for c in contributions
                if float(c["contribution"]) >= 0.08 and float(c["value"]) > 0.0][:3]
    if positive:
        names = "、".join(SIGNAL_LABELS.get(str(c["signal"]), str(c["signal"])) for c in positive)
        parts.append(f"主要依据：{names}。")
    if flow.escalations:
        worst = max(flow.escalations, key=lambda e: e.risk)
        parts.append(f"权限流：{worst.rationale}。")
    if decode.layers:
        kinds = "、".join(sorted({l.kind for l in decode.layers}))
        parts.append(f"检测到 {len(decode.layers)} 层编码（{kinds}），已逐层解码复核。")
    return " ".join(parts)


class Guard:
    """检测护栏主入口。"""

    def __init__(self, model: Optional[GuardModel] = None,
                 policy: Optional[Policy] = None,
                 tracker: Optional[ConversationTracker] = None,
                 deepseek_judge: Optional[Any] = None) -> None:
        self.model = model
        self.policy = policy or default_policy()
        self.policy.validate()
        self.tracker = tracker or ConversationTracker()
        self.deepseek_judge = deepseek_judge

    # ── 输入侧 ──────────────────────────────────────────────────────

    def scan(self, text: str, channel: str = "user",
             conversation_id: str = "") -> Verdict:
        started = time.perf_counter()
        original_len = len(text)
        truncated = False
        if original_len > self.policy.max_input_chars:
            text = text[: self.policy.max_input_chars]
            truncated = True

        try:
            verdict = self._scan_inner(text, channel, conversation_id, started, truncated)
        except Exception as exc:  # noqa: BLE001 —— 失败必须安全
            risk = 1.0 if self.policy.fail_closed else 0.5
            verdict = Verdict(
                risk=risk, action="quarantine" if self.policy.fail_closed else "sanitize",
                channel=channel, model_prob=risk, flow_score=0.0,
                signals={}, contributions=[], evidence=[], decode={}, flow={},
                sanitized=self.policy.data_preface + text,
                removed_segments=[], allowed_capabilities=[],
                explanation=f"分析过程异常，按失败安全策略处理：{exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                truncated=truncated, error=repr(exc),
            )
        return self._maybe_consult_deepseek(verdict, text, channel, started)

    def _maybe_consult_deepseek(self, verdict: Verdict, text: str,
                                channel: str, started: float) -> Verdict:
        """明确结果直接返回；中间动作才请求 DeepSeek 做第二意见。"""
        judge = self.deepseek_judge
        if verdict.error or judge is None:
            return verdict
        if verdict.action not in judge.uncertain_actions:
            verdict.deepseek = {"consulted": False, "reason": "本地规则结果明确"}
            return verdict
        if not judge.configured:
            verdict.deepseek = {"consulted": False, "reason": "DeepSeek 未配置"}
            return verdict

        rule_action = verdict.action
        rule_risk = verdict.risk
        try:
            result = judge.judge(text, channel=channel, rule_risk=rule_risk,
                                 rule_action=rule_action)
        except Exception as exc:  # 外部服务失败时保留本地安全判决
            verdict.deepseek = {
                "consulted": True,
                "used": False,
                "error": str(exc)[:200],
                "reason": "DeepSeek 调用失败，保留本地规则结果",
            }
            verdict.latency_ms = (time.perf_counter() - started) * 1000.0
            return verdict

        info = result.as_dict()
        info.update({"consulted": True, "used": False,
                     "rule_action": rule_action, "rule_risk": round(rule_risk, 4)})
        verdict.deepseek = info
        if result.confidence < judge.min_confidence:
            info["reason"] = (
                f"{result.reason}（置信度低于 {judge.min_confidence:.2f}，保留规则结果）"
            )
            verdict.latency_ms = (time.perf_counter() - started) * 1000.0
            return verdict

        info["used"] = True
        verdict.decision_source = "deepseek"
        verdict.decision_source_label = "DeepSeek"
        if result.is_prompt_injection:
            verdict.action = "block"
            verdict.risk = max(rule_risk, result.confidence)
        else:
            verdict.action = "allow"
            verdict.risk = min(rule_risk, 1.0 - result.confidence)
            verdict.sanitized = text
            verdict.removed_segments = []
        verdict.allowed_capabilities = self.policy.allowed_capabilities(verdict.action)
        verdict.explanation = (
            f"DeepSeek 复核判定：{'提示注入' if result.is_prompt_injection else '非提示注入'}"
            f"（置信度 {result.confidence:.2f}）。{result.reason} "
            f"本地规则原判定为 {rule_action}，风险 {rule_risk:.2f}。"
        )
        verdict.latency_ms = (time.perf_counter() - started) * 1000.0
        return verdict

    def _scan_inner(self, text: str, channel: str, conversation_id: str,
                    started: float, truncated: bool) -> Verdict:
        norm: Normalization = normalize(text)
        folded = norm.folded
        decode: DecodeResult = expand(text)
        seg: Segmentation = segment(text, channel, self.policy.channel_trust)

        # 在"去插空"的紧凑视图上匹配，再把命中位置映射回原文
        compact, index_map = compact_view(folded)
        hits = scan_groups(compact)
        if len(compact) != len(folded):
            hits = remap(hits, index_map)
        flow = analyze(seg, hits)

        # 解码展开后的文本单独再走一遍信息流（编码载荷里的指令在这里现形）
        if decode.expanded:
            extra_text = "\n".join(decode.expanded)
            extra_seg = segment(extra_text, "retrieval")
            extra_flow = analyze(extra_seg, scan_groups(extra_text.lower()))
            flow = _merge_flow(flow, extra_flow)

        signals = features.extract(text, norm, decode, seg, flow, hits)

        seepage = 0.0
        if conversation_id:
            seepage = self.tracker.observe(conversation_id, flow)
        effective_flow = max(flow.score, seepage)
        signals.values["flow_score"] = effective_flow

        if self.model is None:
            raise RuntimeError("模型未加载：请先运行 python -m irongate train")
        lex_text = folded + ("\n" + "\n".join(decode.expanded[:2]) if decode.expanded else "")
        signals = self.model.signals_with_lex(signals, lex_text)
        model_prob = self.model.predict(signals)

        action = self.policy.action_for(model_prob, flow_score=effective_flow,
                                        privilege=flow.max_privilege)
        sanitized, removed = self._sanitize(text, seg, flow, action)
        evidence = self._collect_evidence(hits, seg, decode, flow, norm)
        contributions = self.model.fuser.contributions(signals) if self.model else []

        return Verdict(
            risk=model_prob,
            action=action,
            channel=channel,
            model_prob=model_prob,
            flow_score=effective_flow,
            signals=signals.values,
            contributions=contributions,
            evidence=evidence,
            decode=decode.as_dict(),
            flow=flow.as_dict(),
            sanitized=sanitized,
            removed_segments=removed,
            allowed_capabilities=self.policy.allowed_capabilities(action),
            explanation=_explain(model_prob, action, contributions, flow, decode),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            truncated=truncated,
        )

    # ── 净化 ────────────────────────────────────────────────────────

    def _sanitize(self, text: str, seg: Segmentation, flow: FlowReport,
                  action: str) -> tuple[str, List[int]]:
        """把高危片段替换成占位符，其余内容保留。

        与"整段拒绝"相比，净化能保住上下文可读性（例如检索结果里只有一句
        被污染），与"原样放行"相比，它切断了指令通路。
        """
        if action == "allow":
            return text, []

        risky: Dict[int, float] = {}
        for esc in flow.escalations:
            risky[esc.segment_index] = max(risky.get(esc.segment_index, 0.0), esc.risk)
        for d in flow.directives:
            if d.act in _HIGH_RISK_ACTS and d.strength >= 0.55:
                risky[d.segment_index] = max(risky.get(d.segment_index, 0.0), d.strength * 0.6)
            elif d.act in {"secrecy", "coercion"} and d.strength >= 0.35:
                # 保密/胁迫本身不致命，但它们通常与上面的高危动作成对出现，
                # 净化时一并剥离，避免"别告诉用户"这类尾巴留在放行文本里。
                risky[d.segment_index] = max(risky.get(d.segment_index, 0.0), 0.35)

        if not risky:
            return self.policy.data_preface + text, []

        pieces: List[str] = []
        removed: List[int] = []
        cursor = 0
        n = 0
        for s in seg.segments:
            score = risky.get(s.index, 0.0)
            if score < 0.30:
                continue
            n += 1
            pieces.append(text[cursor:s.start])
            pieces.append(self.policy.isolation_marker.format(n))
            cursor = s.end
            removed.append(s.index)
        pieces.append(text[cursor:])
        body = "".join(pieces)
        return self.policy.data_preface + body, removed

    # ── 证据 ────────────────────────────────────────────────────────

    def _collect_evidence(self, hits: Sequence[Hit], seg: Segmentation,
                          decode: DecodeResult, flow: FlowReport,
                          norm: Normalization) -> List[Evidence]:
        evidence = _evidence_from_hits(hits)
        for m in seg.role_markers:
            evidence.append(Evidence(m.start, m.end, "role_marker", m.term,
                                     "伪造角色/分隔符标记", m.weight))
        for anomaly in norm.anomalies:
            severity = 0.9 if anomaly.kind in {"bidi", "tag"} else 0.65
            evidence.append(Evidence(anomaly.index, anomaly.index + 1,
                                     f"char:{anomaly.kind}", anomaly.name,
                                     f"异常字符 {anomaly.kind}（U+{anomaly.codepoint:04X}）",
                                     severity))
        for layer in decode.layers:
            evidence.append(Evidence(layer.span[0], layer.span[1], f"encoded:{layer.kind}",
                                     layer.kind, "编码载荷，已解码复核",
                                     min(1.0, layer.instruction_score + 0.3)))
        for esc in flow.escalations:
            evidence.append(Evidence(0, 0, "ifc", esc.act, esc.rationale, esc.risk))
        evidence.sort(key=lambda e: (-e.severity, e.start))
        return evidence

    # ── 输出侧 ──────────────────────────────────────────────────────

    def scan_output(self, output: str, *, canary: Optional[str] = None,
                    pending_directives: Iterable[str] = (),
                    allowed_capabilities: Iterable[str] = (),
                    observed_capabilities: Iterable[str] = ()) -> OutputVerdict:
        report = verify_output(
            output, canary=canary, pending_directives=pending_directives,
            allowed_capabilities=allowed_capabilities,
            observed_capabilities=observed_capabilities,
        )
        if report.risk >= self.policy.block_threshold:
            action = "block"
        elif report.risk >= self.policy.sanitize_threshold:
            action = "quarantine"
        else:
            action = "allow"
        return OutputVerdict(report=report, action=action, risk=report.risk)
