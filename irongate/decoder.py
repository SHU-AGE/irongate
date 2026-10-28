"""分层解码：把"藏在编码里的注入"还原出来（原创实现）。

现实中的绕过手法很少靠生僻词，更多是把恶意指令**换个表示法**：
base64、十六进制、URL 编码、Unicode 转义、HTML 实体、ROT13、
甚至用零宽字符承载二进制位流。只看表面文本的检测器会全部漏掉。

本模块的做法：
1. 用一组**保守的**模式找出"看起来像某种编码"的片段；
2. 解码后检查两件事：（a）可打印性；（b）解码结果是否出现指令类词条；
3. 命中则记录一层，并**递归**解码（最多 3 层），因为常见载荷是 "base64(rot13(...))"；
4. 同时尝试整段反转与 ROT13 这类"无需标记"的变换。

只做静态解码，绝不执行任何内容。
"""

from __future__ import annotations

import base64
import binascii
import codecs
import html
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .matching import instruction_pressure, scan_groups

MAX_DEPTH = 3
MAX_LAYERS = 8
MAX_DECODE_LEN = 20_000

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX_RE = re.compile(
    r"(?:\\x[0-9a-fA-F]{2}){4,}"
    r"|(?:0x[0-9a-fA-F]{2}[,\s]*){4,}"
    r"|(?<![0-9a-fA-F])[0-9a-fA-F]{16,}(?![0-9a-fA-F])"
)
_URL_ENC_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")
_UNI_ESC_RE = re.compile(r"(?:\\u[0-9a-fA-F]{4}){5,}")
_HTML_ENT_RE = re.compile(r"(?:&#x?[0-9a-fA-F]{1,6};){4,}")
_ZW_BINARY_RE = re.compile("[\u200b\u200c]{24,}")


@dataclass
class Layer:
    """一层编码。"""

    kind: str
    depth: int
    span: tuple[int, int]
    decoded: str
    instruction_score: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "depth": self.depth,
            "span": list(self.span),
            "decoded_preview": self.decoded[:160],
            "instruction_score": round(self.instruction_score, 3),
        }


@dataclass
class DecodeResult:
    layers: List[Layer] = field(default_factory=list)
    expanded: List[str] = field(default_factory=list)  # 需要一并送检的展开文本
    max_depth: int = 0

    @property
    def depth(self) -> int:
        return self.max_depth

    @property
    def suspicious_layers(self) -> List[Layer]:
        return [l for l in self.layers if l.instruction_score >= 0.35]

    def score(self) -> float:
        """编码风险分：层数越多、解码结果越像指令，分越高。"""
        if not self.layers:
            return 0.0
        base = min(0.55, 0.22 * len(self.layers) + 0.12 * self.max_depth)
        payoff = max(l.instruction_score for l in self.layers)
        return min(1.0, base * (0.45 + 0.75 * payoff))

    def as_dict(self) -> Dict[str, object]:
        return {
            "layer_count": len(self.layers),
            "max_depth": self.max_depth,
            "score": round(self.score(), 3),
            "layers": [l.as_dict() for l in self.layers],
        }


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    good = sum(1 for ch in s if ch.isprintable() or ch in "\n\t ")
    return good / len(s)


def _plausible_text(s: str) -> bool:
    """解码结果是否像"自然语言/代码"而不是随机字节。"""
    if len(s) < 4 or len(s) > MAX_DECODE_LEN:
        return False
    if _printable_ratio(s) < 0.85:
        return False
    letters = sum(1 for ch in s if ch.isalpha() or "\u4e00" <= ch <= "\u9fff")
    return letters / len(s) >= 0.35


def _looks_encoded(text: str) -> float:
    """整段文本本身就是编码体的概率（给"纯 base64 载荷"用）。"""
    compact = text.strip()
    if len(compact) < 16:
        return 0.0
    sym = sum(1 for ch in compact if ch.isalnum() or ch in "+/=_-")
    return sym / len(compact)


def _try_base64(blob: str) -> Optional[bytes]:
    pad = "=" * (-len(blob) % 4)
    try:
        return base64.b64decode(blob + pad, validate=False)
    except (binascii.Error, ValueError):
        return None


def _try_hex(blob: str) -> Optional[bytes]:
    cleaned = re.sub(r"0x|\\x|[\s,:-]", "", blob)
    if len(cleaned) % 2:
        cleaned = cleaned[:-1]
    try:
        return bytes.fromhex(cleaned)
    except ValueError:
        return None


def _decoded_instruction_score(text: str) -> float:
    hits = scan_groups(text, ("override", "role", "authority", "exfil", "capability"))
    return instruction_pressure(hits)


def _record(kind: str, depth: int, span: tuple[int, int], decoded: str,
            out: List[Layer], queue: List[str]) -> None:
    layer = Layer(kind, depth, span, decoded, _decoded_instruction_score(decoded))
    out.append(layer)
    if layer.instruction_score >= 0.2:
        queue.append(decoded)


def _decode_pass(text: str, depth: int, out: List[Layer], queue: List[str]) -> None:
    """在單层文本上找出所有编码片段并解码。"""
    if depth > MAX_DEPTH or len(out) >= MAX_LAYERS:
        return

    for m in _BASE64_RE.finditer(text):
        raw = _try_base64(m.group(0))
        if not raw:
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if _plausible_text(decoded):
            _record("base64", depth, (m.start(), m.end()), decoded, out, queue)

    for m in _HEX_RE.finditer(text):
        raw = _try_hex(m.group(0))
        if not raw:
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if _plausible_text(decoded):
            _record("hex", depth, (m.start(), m.end()), decoded, out, queue)

    for m in _URL_ENC_RE.finditer(text):
        decoded = urllib.parse.unquote(m.group(0))
        if _plausible_text(decoded) and decoded != m.group(0):
            _record("url", depth, (m.start(), m.end()), decoded, out, queue)

    for m in _UNI_ESC_RE.finditer(text):
        try:
            decoded = codecs.decode(m.group(0), "unicode_escape")
        except (UnicodeDecodeError, ValueError):
            continue
        if _plausible_text(decoded):
            _record("unicode_escape", depth, (m.start(), m.end()), decoded, out, queue)

    for m in _HTML_ENT_RE.finditer(text):
        decoded = html.unescape(m.group(0))
        if _plausible_text(decoded) and decoded != m.group(0):
            _record("html_entity", depth, (m.start(), m.end()), decoded, out, queue)

    for m in _ZW_BINARY_RE.finditer(text):
        bits = "".join("0" if ch == "\u200b" else "1" for ch in m.group(0))
        by = bytearray()
        for i in range(0, len(bits) - 7, 8):
            by.append(int(bits[i:i + 8], 2))
        try:
            decoded = by.decode("utf-8")
        except UnicodeDecodeError:
            decoded = ""
        if _plausible_text(decoded):
            _record("zero_width_binary", depth, (m.start(), m.end()), decoded, out, queue)


def _hidden_transforms(text: str, out: List[Layer], queue: List[str]) -> None:
    """无标记变换：整段反转、整段 ROT13。

    只在**变换后**才出现指令类词条时才记录，避免把正常文本误判成编码。
    """
    if len(text) < 8:
        return
    base = _decoded_instruction_score(text)

    reversed_text = text[::-1]
    if _decoded_instruction_score(reversed_text) > max(0.3, base + 0.2):
        _record("reversed", 1, (0, len(text)), reversed_text, out, queue)

    rot = codecs.encode(text, "rot13")
    if _decoded_instruction_score(rot) > max(0.3, base + 0.2):
        _record("rot13", 1, (0, len(text)), rot, out, queue)


def expand(text: str, *, max_depth: int = MAX_DEPTH) -> DecodeResult:
    """逐层解码，返回所有层与需要送检的展开文本。"""
    result = DecodeResult()
    if not text:
        return result

    frontier: List[tuple[str, int]] = [(text, 0)]
    seen = {text}

    while frontier:
        current, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        layers: List[Layer] = []
        queue: List[str] = []
        _decode_pass(current, depth, layers, queue)
        if depth == 0:
            _hidden_transforms(current, layers, queue)
        result.layers.extend(layers)
        result.max_depth = max(result.max_depth, depth + (1 if layers else 0))
        for txt in queue:
            if txt in seen:
                continue
            seen.add(txt)
            result.expanded.append(txt)
            frontier.append((txt, depth + 1))
        if len(result.layers) >= MAX_LAYERS:
            break

    # 纯编码体（整段就是一个 base64 blob）单独兜底
    if not result.layers and _looks_encoded(text) > 0.98 and len(text.strip()) >= 24:
        raw = _try_base64(text.strip())
        if raw:
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                decoded = ""
            if _plausible_text(decoded):
                _record("base64_whole", 0, (0, len(text)), decoded, result.layers, [])
                result.max_depth = 1
                result.expanded.append(decoded)

    return result
