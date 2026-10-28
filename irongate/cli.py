"""命令行入口：`python -m irongate <command>`。

命令刻意保持最少，覆盖比赛现场的五个动作：
    复现训练 → 看指标 → 现场扫一段文本 → 起一个可交互的演示页 → 查看生效配置

配置来源优先级：**命令行参数 > 环境变量 / `.env` > 内置默认值**。
`.env` 是可选便利项；不建它也能用。
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__, config
from .guard import Guard
from .deepseek import DeepSeekJudge
from .pipeline import build_dataset, evaluate_set, load_model, train
from .policy import Policy, PolicyError

_ACTION_COLOR = {
    "allow": "\033[32m",       # 绿
    "sanitize": "\033[33m",    # 黄
    "quarantine": "\033[35m",  # 紫
    "block": "\033[31m",       # 红
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _color(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled else text


def _policy_for(settings: config.Settings) -> Policy:
    """读取策略文件（若配置了），并用环境变量覆盖运行期开关。"""
    if settings.policy_path and settings.policy_path.exists():
        policy = Policy.load(str(settings.policy_path))
    else:
        policy = Policy()
    policy.fail_closed = settings.fail_closed
    policy.max_input_chars = settings.max_input_chars
    policy.validate()
    return policy


def _model_and_policy(settings: config.Settings) -> Tuple[object, Policy]:
    return load_model(settings.model_path), _policy_for(settings)


def _deepseek_judge(settings: config.Settings) -> DeepSeekJudge:
    return DeepSeekJudge(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        timeout_seconds=settings.deepseek_timeout_seconds,
        min_confidence=settings.deepseek_min_confidence,
        enabled=settings.deepseek_enabled,
    )


def _print_verdict(verdict, *, as_json: bool, color: bool) -> None:
    payload = verdict.as_dict()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    action = payload["action"]
    colored = _color(action.upper(), _ACTION_COLOR.get(action, ""), color)
    print(f"{_color('动作', _BOLD, color)}: {colored}")
    print(f"{_color('判定来源', _BOLD, color)}: {payload['decision_source_label']}")
    print(f"{_color('风险', _BOLD, color)}: {payload['risk']:.3f}   "
          f"模型 {payload['model_prob']:.3f}   权限流 {payload['flow_score']:.3f}")
    print(f"{_color('解释', _BOLD, color)}: {payload['explanation']}")
    contributions = payload["contributions"]
    active = [c for c in contributions if c["value"] > 0 and c["contribution"] >= 0.05][:4]
    suppressors = [c for c in contributions if c["contribution"] <= -0.10][:2]
    if active:
        print(f"{_color('主要信号', _BOLD, color)}（贡献 = 权重 × 相对训练基线的标准化偏离）：")
        for c in active:
            print(f"  · {c['signal']:<18} 取值 {c['value']:<7} 贡献 +{c['contribution']}")
    if suppressors:
        print(f"{_color('抑制项', _BOLD, color)}：")
        for c in suppressors:
            print(f"  · {c['signal']:<18} 贡献 {c['contribution']}")
    if payload["decode"]["layer_count"]:
        print(f"{_color('编码层', _BOLD, color)}: "
              + ", ".join(l["kind"] for l in payload["decode"]["layers"]))
    if payload["flow"]["escalations"]:
        worst = max(payload["flow"]["escalations"], key=lambda e: e["risk"])
        print(f"{_color('越界', _BOLD, color)}: {worst['rationale']}")
    if payload["action"] != "allow":
        print(f"{_color('净化后', _BOLD, color)}: {payload['sanitized'][:300]}")
    print(f"{_color('耗时', _BOLD, color)}: {payload['latency_ms']} ms")


def _cmd_scan(args: argparse.Namespace) -> int:
    model, policy = _model_and_policy(args.settings)
    guard = Guard(model=model, policy=policy,
                  deepseek_judge=_deepseek_judge(args.settings))
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        print('没有输入文本（用 `irongate scan "..."` 或 --file）', file=sys.stderr)
        return 2
    verdict = guard.scan(text, channel=args.channel, conversation_id=args.conversation)
    color = sys.stdout.isatty() and not args.no_color
    _print_verdict(verdict, as_json=args.json, color=color)
    if args.strict and verdict.action in {"block", "quarantine"}:
        return 1
    return 0


def _cmd_scan_output(args: argparse.Namespace) -> int:
    model, policy = _model_and_policy(args.settings)
    guard = Guard(model=model, policy=policy)
    canary = args.canary or args.settings.canary
    result = guard.scan_output(
        args.text,
        canary=canary,
        allowed_capabilities=args.allowed.split(",") if args.allowed else (),
        observed_capabilities=args.observed.split(",") if args.observed else (),
    )
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"动作: {result.action.upper()}  风险: {result.risk:.3f}")
        if result.report.notes:
            for note in result.report.notes:
                print(f"  · {note}")
    return 1 if result.action == "block" else 0


def _cmd_train(args: argparse.Namespace) -> int:
    out_path = Path(args.out) if args.out else args.settings.model_path
    dataset = build_dataset(n_attack=args.attack, n_benign=args.benign, seed=args.seed)
    model, info = train(dataset, out_path=out_path)
    print(f"训练完成：{info['train_size']} 条样本，{info['train_seconds']} 秒，"
          f"词表 {info['nb_vocab']} 项")
    print(f"模型已写入 {info['model_path']}")
    for name, split in (("dev", dataset.dev), ("test", dataset.test)):
        report = evaluate_set(name, model, split)
        print(report.row())
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .bench import run_benchmark

    model = load_model(args.settings.model_path)
    result = run_benchmark(model=model, seed=args.seed, folds=args.folds)
    if args.json:
        Path(args.json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {args.json}")
    else:
        print(result["markdown"])
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .api import serve

    host = args.host or args.settings.host
    port = args.port or args.settings.port
    url = f"http://{host}:{port}"
    print(f"IronGate 演示服务启动：{url}")
    print("按 Ctrl+C 停止")
    if args.open:
        webbrowser.open(url)
    serve(host=host, port=port)
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    payload = args.settings.describe()
    payload["dotenv_loaded"] = bool(args.dotenv_loaded)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key:<18} {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="irongate",
        description="IronGate（铁闸）—— 大模型提示注入检测与护栏（原创实现）",
    )
    parser.add_argument("--version", action="version", version=f"irongate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="检测一段文本")
    scan.add_argument("text", nargs="?", default="", help="要检测的文本")
    scan.add_argument("--file", help="从文件读取文本")
    scan.add_argument("--channel", default="user",
                      choices=["system", "user", "tool", "retrieval", "unknown"])
    scan.add_argument("--conversation", default="", help="会话 id（启用多轮渗流追踪）")
    scan.add_argument("--model", help="模型 JSON 路径（默认取 IRONGATE_MODEL）")
    scan.add_argument("--policy", help="策略 JSON 路径（默认取 IRONGATE_POLICY）")
    scan.add_argument("--json", action="store_true")
    scan.add_argument("--no-color", action="store_true")
    scan.add_argument("--strict", action="store_true", help="拦截/隔离时返回退出码 1")
    scan.set_defaults(func=_cmd_scan)

    out = sub.add_parser("scan-output", help="校验模型输出（金丝雀/回响/能力越界）")
    out.add_argument("text")
    out.add_argument("--canary", default=None,
                     help="系统提示中埋入的金丝雀串（默认取 IRONGATE_CANARY）")
    out.add_argument("--allowed", default="", help="授权能力，逗号分隔")
    out.add_argument("--observed", default="", help="实际调用能力，逗号分隔")
    out.add_argument("--model", help="模型 JSON 路径")
    out.add_argument("--json", action="store_true")
    out.set_defaults(func=_cmd_scan_output)

    tr = sub.add_parser("train", help="生成合成语料并训练模型")
    tr.add_argument("--attack", type=int, default=2200)
    tr.add_argument("--benign", type=int, default=2200)
    tr.add_argument("--seed", type=int, default=20260101)
    tr.add_argument("--out", default=None, help="模型输出路径")
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="跑完整评测（交叉验证/消融/鲁棒性）")
    ev.add_argument("--seed", type=int, default=20260101)
    ev.add_argument("--folds", type=int, default=5)
    ev.add_argument("--json", default="")
    ev.set_defaults(func=_cmd_eval)

    sv = sub.add_parser("serve", help="启动本地演示服务")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--open", action="store_true", help="自动打开浏览器")
    sv.set_defaults(func=_cmd_serve)

    demo = sub.add_parser("demo", help="等价于 serve --open")
    demo.add_argument("--host", default=None)
    demo.add_argument("--port", type=int, default=None)
    demo.set_defaults(func=_cmd_serve, open=True)

    cfg = sub.add_parser("config", help="打印当前生效的配置来源")
    cfg.add_argument("--json", action="store_true")
    cfg.set_defaults(func=_cmd_config)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        loaded = config.load_dotenv()
        settings = config.settings_from_env()
        if getattr(args, "model", None):
            settings = replace(settings, model_path=Path(args.model))
        if getattr(args, "policy", None):
            settings = replace(settings, policy_path=Path(args.policy))
        args.settings = settings
        args.dotenv_loaded = sorted(loaded)
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except PolicyError as exc:
        print(f"策略错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
