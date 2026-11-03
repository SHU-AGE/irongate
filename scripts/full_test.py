"""完整测试：用项目里的全部测试样本逐条走一遍真实判定流程（原创，仅标准库）。

用法：
    python scripts/full_test.py                          # 结果写到 reports/完整测试明细_<日期>.csv
    python scripts/full_test.py --output my.csv
    python scripts/full_test.py --no-deepseek            # 只用本地规则，不调用 API

样本（与 reports/完整测试报告 一致）：
1. bench/challenges.jsonl        人工困难样本（攻击 + 正常）；
2. bench/benign_realistic.jsonl  真实风格的正常样本；
3. 生成器留出测试集              按"文本 + 通道"去重，按模板家族轮流挑，攻击/正常各最多 100 条。

DeepSeek 配置从 .env 读取（见 .env.example）；没有 Key 时自动退回只用本地规则。
不会打印或写出 API Key。
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from irongate import config  # noqa: E402
from irongate.cli import _deepseek_judge, _model_and_policy  # noqa: E402
from irongate.guard import Guard  # noqa: E402
from irongate.pipeline import build_dataset  # noqa: E402

ACT = {"allow": "放行", "sanitize": "净化", "quarantine": "隔离", "block": "拦截"}
PER_LABEL_LIMIT = 100


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_samples():
    samples = []
    for r in _read_jsonl(PROJECT_ROOT / "bench" / "challenges.jsonl"):
        samples.append({"source": "人工困难样本", "label": r["label"], "type": r["kind"],
                        "channel": r.get("channel", "user"), "text": r["text"]})
    for r in _read_jsonl(PROJECT_ROOT / "bench" / "benign_realistic.jsonl"):
        samples.append({"source": "真实风格正常样本", "label": 0, "type": r["kind"],
                        "channel": r.get("channel", "user"), "text": r["text"]})

    test = build_dataset().test
    for label in (1, 0):
        by_family = collections.OrderedDict()
        seen = set()
        for s in test:
            key = (s.text, s.channel)
            if s.label != label or key in seen:
                continue
            seen.add(key)
            by_family.setdefault(s.family.split("#")[0], []).append(s)
        picked = []
        while len(picked) < PER_LABEL_LIMIT and any(by_family.values()):
            for fam, items in by_family.items():
                if items and len(picked) < PER_LABEL_LIMIT:
                    picked.append((fam, items.pop(0)))
        for fam, s in picked:
            samples.append({"source": "生成器样本", "label": label, "type": fam,
                            "channel": s.channel, "text": s.text})
    return samples


def run(guard: Guard, samples):
    rows = []
    started = time.time()
    for i, s in enumerate(samples, 1):
        v = guard.scan(s["text"], channel=s["channel"])
        ds = v.deepseek or {}
        rows.append({**s,
                     "rule_action": ds.get("rule_action", v.action),
                     "rule_risk": ds.get("rule_risk", round(v.risk, 4)),
                     "final_action": v.action,
                     "decision_source": v.decision_source,
                     "ds_consulted": bool(ds.get("consulted")),
                     "ds_used": bool(ds.get("used")),
                     "ds_failed": bool(ds.get("consulted")) and "error" in ds,
                     "ds_injection": ds.get("is_prompt_injection"),
                     "ds_confidence": ds.get("confidence"),
                     "ds_reason": ds.get("reason") or ds.get("error") or ""})
        if i % 50 == 0:
            print(f"已跑 {i}/{len(samples)}，用时 {time.time() - started:.0f} 秒", flush=True)
    return rows


def stopped(action: str) -> bool:
    return action != "allow"


def pct(k: int, n: int) -> str:
    return f"{k}/{n}（{k / n * 100:.1f}%）" if n else "-"


def summarize(rows) -> str:
    lines = []
    attacks = [r for r in rows if r["label"] == 1]
    normals = [r for r in rows if r["label"] == 0]
    lines.append(f"样本 {len(rows)} 条：攻击 {len(attacks)}，正常 {len(normals)}")
    lines.append("")
    lines.append("| 样本来源 | 攻击拦下（仅本地） | 攻击拦下（最终） | 正常误拦（仅本地） | 正常误拦（最终） |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    sources = list(dict.fromkeys(r["source"] for r in rows)) + ["合计"]
    for src in sources:
        group = rows if src == "合计" else [r for r in rows if r["source"] == src]
        a = [r for r in group if r["label"] == 1]
        n = [r for r in group if r["label"] == 0]
        lines.append(
            f"| {src} | {pct(sum(stopped(r['rule_action']) for r in a), len(a))} "
            f"| {pct(sum(stopped(r['final_action']) for r in a), len(a))} "
            f"| {pct(sum(stopped(r['rule_action']) for r in n), len(n))} "
            f"| {pct(sum(stopped(r['final_action']) for r in n), len(n))} |")
    consulted = [r for r in rows if r["ds_consulted"]]
    lines.append("")
    lines.append(f"DeepSeek 复核 {len(consulted)} 条：采纳 {sum(r['ds_used'] for r in consulted)}，"
                 f"置信度不足未采纳 {sum(not r['ds_used'] and not r['ds_failed'] for r in consulted)}，"
                 f"调用失败 {sum(r['ds_failed'] for r in consulted)}")
    lines.append("")
    lines.append("按攻击类型（最终）：")
    by_type = collections.defaultdict(list)
    for r in attacks:
        by_type[r["type"]].append(r)
    for t, group in sorted(by_type.items()):
        lines.append(f"  {t:<22} {pct(sum(stopped(r['final_action']) for r in group), len(group))}")
    wrong = [r for r in rows if stopped(r["final_action"]) != (r["label"] == 1)]
    lines.append("")
    lines.append(f"判错 {len(wrong)} 条：")
    for r in wrong:
        kind = "漏拦" if r["label"] == 1 else "误拦"
        lines.append(f"  [{kind}] 本地{ACT[r['rule_action']]}({r['rule_risk']}) → 最终{ACT[r['final_action']]}"
                     f" | {r['text'][:60]}")
    return "\n".join(lines)


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # 带 BOM，Excel/Numbers 直接打开不乱码
        w = csv.writer(f)
        w.writerow(["序号", "来源", "真实标签", "类型", "来源通道", "句子", "本地判定", "本地风险",
                    "最终判定", "判定来源", "是否判对", "DeepSeek是否复核", "DeepSeek认为是注入",
                    "DeepSeek置信度", "DeepSeek理由"])
        for i, r in enumerate(rows, 1):
            ok = stopped(r["final_action"]) == (r["label"] == 1)
            w.writerow([i, r["source"], "攻击" if r["label"] else "正常", r["type"], r["channel"],
                        r["text"].replace("​", "[零宽]"), ACT[r["rule_action"]], r["rule_risk"],
                        ACT[r["final_action"]], "DeepSeek" if r["decision_source"] == "deepseek" else "本地",
                        "对" if ok else "错", "是" if r["ds_consulted"] else "否",
                        {True: "是", False: "否"}.get(r["ds_injection"], ""),
                        r["ds_confidence"] if r["ds_consulted"] else "", r["ds_reason"] if r["ds_consulted"] else ""])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="IronGate 完整测试（本地规则 + DeepSeek 复核）")
    parser.add_argument("--output", default=str(
        PROJECT_ROOT / "reports" / f"完整测试明细_{dt.date.today().isoformat()}.csv"))
    parser.add_argument("--no-deepseek", action="store_true", help="只用本地规则，不调用 DeepSeek")
    args = parser.parse_args(argv)

    settings = config.load_settings()
    model, policy = _model_and_policy(settings)
    judge = None if args.no_deepseek else _deepseek_judge(settings)
    if judge is not None and not judge.configured:
        print("提示：没有读到 DeepSeek Key，本次只用本地规则。")
        judge = None
    else:
        print("DeepSeek 复核：" + (f"开启（模型 {settings.deepseek_model}）" if judge else "关闭"))
    guard = Guard(model=model, policy=policy, deepseek_judge=judge)

    rows = run(guard, load_samples())
    write_csv(rows, Path(args.output))
    print()
    print(summarize(rows))
    print()
    print(f"逐条明细已写入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
