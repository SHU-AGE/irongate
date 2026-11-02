"""运行配置：`.env` 文件与 IRONGATE_* 环境变量（原创，零依赖）。

为什么自己解析 `.env`
--------------------
引入 `python-dotenv` 只为一个几十行的解析器，会破坏"零依赖"这条底线。
这里实现的是**保守子集**：`KEY=VALUE`、`#` 注释、可选的单双引号包裹、
忽略空行与非法行；已存在的进程环境变量优先，不被文件覆盖。

解析失败永远不抛异常——配置缺失时退回默认值，比"起不来"更符合护栏的定位。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "model.json"

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def parse_env_file(path: Path) -> Dict[str, str]:
    """解析 .env 文件，返回键值对；文件不存在或不可读时返回空字典。"""
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: Optional[Path] = None, *, environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """把 .env 载入进程环境（不覆盖已有变量）。返回实际载入的键值对。"""
    target = Path(path) if path else PROJECT_ROOT / ".env"
    env = environ if environ is not None else os.environ
    loaded: Dict[str, str] = {}
    for key, value in parse_env_file(target).items():
        if key in env:
            continue
        env[key] = value
        loaded[key] = value
    return loaded


def _as_bool(value: str, default: bool) -> bool:
    low = value.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    return default


def _as_int(value: str, default: int, low: int, high: int) -> int:
    try:
        number = int(value.strip())
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _as_float(value: str, default: float, low: float, high: float) -> float:
    try:
        number = float(value.strip())
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


@dataclass(frozen=True)
class Settings:
    """从环境变量解析出的运行参数。"""

    model_path: Path
    policy_path: Optional[Path]
    host: str
    port: int
    fail_closed: bool
    max_input_chars: int
    canary: Optional[str]
    deepseek_enabled: bool
    deepseek_api_key: Optional[str]
    deepseek_base_url: str
    deepseek_model: str
    deepseek_timeout_seconds: int
    deepseek_min_confidence: float

    def describe(self) -> Dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "policy_path": str(self.policy_path) if self.policy_path else None,
            "host": self.host,
            "port": self.port,
            "fail_closed": self.fail_closed,
            "max_input_chars": self.max_input_chars,
            "canary": "***" if self.canary else None,
            "deepseek_enabled": self.deepseek_enabled,
            "deepseek_configured": bool(self.deepseek_api_key),
            "deepseek_api_key": "***" if self.deepseek_api_key else None,
            "deepseek_base_url": self.deepseek_base_url,
            "deepseek_model": self.deepseek_model,
            "deepseek_timeout_seconds": self.deepseek_timeout_seconds,
            "deepseek_min_confidence": self.deepseek_min_confidence,
        }


def settings_from_env(environ: Optional[Mapping[str, str]] = None,
                      root: Path = PROJECT_ROOT) -> Settings:
    env = environ if environ is not None else os.environ

    def path_of(key: str, default: Optional[Path]) -> Optional[Path]:
        raw = (env.get(key) or "").strip()
        if not raw:
            return default
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else (root / candidate)

    model = path_of("IRONGATE_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    policy = path_of("IRONGATE_POLICY", None)
    host = (env.get("IRONGATE_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = _as_int(env.get("IRONGATE_PORT", ""), 8100, 1, 65535)
    fail_closed = _as_bool(env.get("IRONGATE_FAIL_CLOSED", ""), True)
    max_chars = _as_int(env.get("IRONGATE_MAX_INPUT_CHARS", ""), 40000, 256, 200000)
    canary = (env.get("IRONGATE_CANARY") or "").strip() or None
    deepseek_enabled = _as_bool(env.get("IRONGATE_DEEPSEEK_ENABLED", ""), True)
    deepseek_api_key = (env.get("DEEPSEEK_API_KEY") or "").strip() or None
    deepseek_base_url = (env.get("DEEPSEEK_BASE_URL") or
                         "https://api.deepseek.com").strip().rstrip("/")
    deepseek_model = (env.get("DEEPSEEK_MODEL") or "deepseek-flash").strip()
    deepseek_timeout = _as_int(env.get("IRONGATE_DEEPSEEK_TIMEOUT", ""), 20, 1, 120)
    deepseek_min_confidence = _as_float(
        env.get("IRONGATE_DEEPSEEK_MIN_CONFIDENCE", ""), 0.65, 0.5, 1.0)
    return Settings(model_path=model, policy_path=policy, host=host, port=port,
                    fail_closed=fail_closed, max_input_chars=max_chars, canary=canary,
                    deepseek_enabled=deepseek_enabled,
                    deepseek_api_key=deepseek_api_key,
                    deepseek_base_url=deepseek_base_url,
                    deepseek_model=deepseek_model,
                    deepseek_timeout_seconds=deepseek_timeout,
                    deepseek_min_confidence=deepseek_min_confidence)


def load_settings(dotenv: bool = True) -> Settings:
    """CLI/服务启动时的统一入口。"""
    if dotenv:
        load_dotenv()
    return settings_from_env()
