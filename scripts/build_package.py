"""打包脚本：生成可提交的 IronGate 压缩包（原创，仅标准库）。

用法：
    python scripts/build_package.py                       # 默认输出到 dist/IronGate_v1.0.zip
    python scripts/build_package.py --output "D:\\IronGate_Project\\IronGate_v1.0.zip"

行为：
1. 目标目录不存在时用 `os.makedirs` 递归创建（含父目录）；
2. 打包前做排除：`.env` 系列（保留 `.env.example`）、`__pycache__`、`.venv`、
   `*.pyc`、`*.pyo`、`.git`、`*.log` 等；
3. 若根目录缺少 `requirements.txt` / `.env.example` 会先补齐（防止手工删掉后打出残缺包）；
4. 打印清单摘要：收录文件数、排除项、总体积、顶层条目，便于核对。
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    "__pycache__", ".venv", "venv", ".git", ".pytest_cache", ".mypy_cache",
    ".idea", ".vscode", ".tmp_config", "node_modules", "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".bak"}
EXCLUDED_FILES = {".DS_Store", "Thumbs.db", ".env", ".env.local", ".env.development",
                  ".env.production", ".env.test"}

REQUIRED_ROOT_FILES = ("requirements.txt", ".env.example")

DEFAULT_REQUIREMENTS = """# IronGate 没有第三方运行时依赖：只使用 Python 标准库（>= 3.10）。
# 保留本文件是为了说明"解压即可运行、不需要联网装依赖"。
"""

DEFAULT_ENV_EXAMPLE = """# IronGate 配置模板：复制为 .env 后按需修改（.env 不会被提交，也不会被打进压缩包）
#   macOS / Linux:  cp .env.example .env
#   Windows:        copy .env.example .env
# 所有项都可以留空，留空即使用内置默认值。

# —— 本地引擎 ——
IRONGATE_MODEL=artifacts/model.json
IRONGATE_POLICY=
IRONGATE_FAIL_CLOSED=true
IRONGATE_MAX_INPUT_CHARS=40000
IRONGATE_HOST=127.0.0.1
IRONGATE_PORT=8100
IRONGATE_CANARY=

# —— DeepSeek 复核（可选；不填 Key 时只用本地规则）——
IRONGATE_DEEPSEEK_ENABLED=true
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
IRONGATE_DEEPSEEK_TIMEOUT=20
IRONGATE_DEEPSEEK_MIN_CONFIDENCE=0.65
"""


def is_excluded(rel: Path) -> bool:
    """按相对路径判断是否排除。"""
    parts = rel.parts
    if any(part in EXCLUDED_DIRS for part in parts[:-1]):
        return True
    name = rel.name
    if name in EXCLUDED_FILES:
        return True
    # 任意 *.env 都排除，但保留模板 .env.example
    if name.endswith(".env") or (name.startswith(".env.") and name != ".env.example"):
        return True
    if rel.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def ensure_root_files(root: Path, log) -> None:
    req = root / "requirements.txt"
    if not req.exists():
        req.write_text(DEFAULT_REQUIREMENTS, encoding="utf-8")
        log(f"补齐缺失文件：{req.name}")
    env_example = root / ".env.example"
    if not env_example.exists():
        env_example.write_text(DEFAULT_ENV_EXAMPLE, encoding="utf-8")
        log(f"补齐缺失文件：{env_example.name}")


def build(root: Path, output: Path, log=print) -> dict:
    ensure_root_files(root, log)

    output.parent.mkdir(parents=True, exist_ok=True)  # 等价于 os.makedirs(parents=True, exist_ok=True)

    included: list[Path] = []
    excluded: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        if is_excluded(rel):
            excluded.append(rel)
        else:
            included.append(rel)

    tmp_target = output.with_suffix(output.suffix + ".part")
    with zipfile.ZipFile(tmp_target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for rel in included:
            archive.write(root / rel, arcname=str(Path(root.name) / rel))
    os.replace(tmp_target, output)

    size = output.stat().st_size
    log(f"打包完成：{output}")
    log(f"  收录 {len(included)} 个文件，排除 {len(excluded)} 个")
    log(f"  体积 {size / 1024:.1f} KB")
    if excluded:
        shown = ", ".join(str(p) for p in excluded[:6])
        more = "" if len(excluded) <= 6 else f" …（其余 {len(excluded) - 6} 项）"
        log(f"  已排除：{shown}{more}")
    for name in REQUIRED_ROOT_FILES:
        rel = Path(name)
        log(f"  {name}: {'已包含' if rel in included else '缺失！'}")
    return {"included": len(included), "excluded": len(excluded), "size": size,
            "output": str(output)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="打包 IronGate 工程")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="工程根目录")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "dist" / "IronGate_v1.0.zip"),
                        help="输出 zip 路径")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if not (root / "irongate").is_dir():
        print(f"错误：{root} 不像 IronGate 工程根目录（缺少 irongate/ 包）", file=sys.stderr)
        return 2
    result = build(root, Path(args.output))
    return 0 if result["included"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
