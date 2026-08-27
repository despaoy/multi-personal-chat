"""月社妃游戏原文逐行状态机解析器（P2/P2.1）。

替换旧的跨文件全文 DOTALL 正则（scripts/extract_character_dialogues.py 的 SCRIPT_RE），
修复 7 处未闭合开引号导致的台词吞并问题，同时保留 9 条合法跨行台词。

核心规则（详见 docs/research/KISAKI_GAME_RAG_PARSER.md）：
- 只用行首锚定的正则识别「说话人标签 + 开引号」，绝不用正则跨行吞取台词内容；
- 台词跨行时逐行累积，直到对应闭引号出现；闭引号后只允许空白或同类闭引号的重复
  （重复属原文排版瑕疵，保留进文本并告警 duplicate_closing_quote），
  其他非空内容以稳定错误码 trailing_content_after_quote 显式失败，绝不静默丢弃；
- 未闭合台词在遇到下一条说话人标签行或 EOF 时截断，保留原文并写入
  unclosed_quote 告警，绝不静默吞并下一条台词；
- 叙述按连续物理行合并，空行只结束当前叙述块，不生成空 narration；
- 换行规范化：CRLF/CR 统一为 LF，这是唯一允许的文本规范化；
- 溯源与 ID 均基于便携 source_path（相对 POSIX 路径），不依赖机器绝对路径；
  段 ID 由 sha256(source_path|segment_type|line_start|line_end) 派生，
  不含段序号，前面无关段落的切分变化不会波及后续段 ID。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from knowledge.game_rag.models import QuoteStyle, ScriptSegment, SegmentType, SourceSpan

if TYPE_CHECKING:
    from pathlib import Path

# 行首锚定的说话人标签：前导空白 + [说话人] + 空白 + 开引号（三种之一）。
# 只负责"识别台词起点"，不负责提取台词内容。
_TAG_LINE_RE = re.compile(r"^\s*\[([^\[\]\n]+)\]\s*([「『“])")

# 开引号 -> (对应闭引号, 引号样式)
_OPEN_QUOTES: dict[str, tuple[str, QuoteStyle]] = {
    "「": ("」", QuoteStyle.corner),
    "『": ("』", QuoteStyle.double_corner),
    "“": ("”", QuoteStyle.curly),
}

# 稳定告警码：闭引号后存在非空内容（当前模型无法表达同一物理行上的两个不重叠段）
TRAILING_CONTENT_AFTER_QUOTE = "trailing_content_after_quote"

# 稳定告警码：闭引号后仅跟随同类闭引号的重复（原文排版瑕疵，如 12青金石:2540 的 」」）。
# 重复字符保留进 text（不丢数据、不改原文），并写入告警供审计。
DUPLICATE_CLOSING_QUOTE = "duplicate_closing_quote"

# 稳定告警码：文件末行是单独的 DOS EOF 标记（SUB / 0x1A，旧式 Ctrl-Z 文件结尾，
# 见 日后谈.txt:879）。仅当末行内容**严格等于** \x1a 时忽略该行（编辑器产物，
# 非剧本内容，保留会生成 text 为控制字符的 narration 段）；其余位置的控制字符
# 一律按原文保留，不做全局控制字符清洗，也不改写源文件。
DOS_EOF_MARKER = "dos_eof_marker"


class ScriptParseError(ValueError):
    """解析器显式失败：携带便携 source_path、物理行号与稳定错误码。"""

    def __init__(self, source_path: str, line_no: int, code: str, message: str):
        self.source_path = source_path
        self.line_no = line_no
        self.code = code
        super().__init__(f"{code}: {message} (source_path={source_path}, line_no={line_no})")


def unclosed_quote_warning(next_speaker_line: int | None, file_end_line: int) -> str:
    """未闭合引号告警的集中定义，保证格式稳定可测试。

    - next_speaker_line 非 None：在下一条说话人标签行（第 N 行）前截断；
    - next_speaker_line 为 None：一直未闭合直到文件结束。
    """
    if next_speaker_line is not None:
        return f"unclosed_quote: next_speaker_line={next_speaker_line}"
    return f"unclosed_quote: next_speaker_line=EOF(file_end_line={file_end_line})"


def _stable_segment_id(source_path: str, segment_type: SegmentType, line_start: int, line_end: int) -> str:
    """确定性段 ID：仅由便携 source_path、segment_type 与 SourceSpan 派生。

    刻意不含段序号：同文件内 SourceSpan 互不重叠，(segment_type, span) 已唯一；
    去掉序号后，前面无关段落的切分变化不会改变后续段的 ID。
    """
    digest = hashlib.sha256(f"{source_path}|{segment_type.value}|{line_start}|{line_end}".encode()).hexdigest()
    return f"seg_{digest[:16]}"


def dos_eof_marker_warning() -> str:
    """DOS EOF 标记告警的集中定义，保证格式稳定可测试。"""
    return f"{DOS_EOF_MARKER}: trailing standalone 0x1A ignored"


def _split_lines_with_eof_flag(text: str) -> tuple[list[str], bool]:
    """按物理行切分并检测末行 DOS EOF 标记，返回 (行列表, 末行是否为单独的 \\x1a)。

    末行严格等于 \\x1a 时移除该行：它是旧式 DOS EOF 标记（编辑器产物）而非剧本
    内容。除此之外不做任何控制字符清洗；行中/行尾/非末行出现的 \\x1a 一律按原文
    保留。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    has_dos_eof = bool(lines) and lines[-1] == "\x1a"
    if has_dos_eof:
        lines.pop()
    return lines, has_dos_eof


def _split_lines(text: str) -> list[str]:
    """按物理行切分：CRLF/CR 规范化为 LF，去掉文件末尾换行产生的哨兵空串；
    末行若为单独的 DOS EOF 标记（\\x1a）同样移除。"""
    return _split_lines_with_eof_flag(text)[0]


def duplicate_closing_quote_warning(count: int) -> str:
    """重复闭引号告警的集中定义，保证格式稳定可测试。"""
    return f"{DUPLICATE_CLOSING_QUOTE}: count={count}"


def _after_close_resolution(
    source_path: str,
    line_no: int,
    after_close: str,
    close_char: str,
) -> tuple[str, list[str]]:
    """闭引号后内容判定，返回 (需追加保留的原文, 告警列表)。

    - 纯空白：正常结束；
    - 仅由同类闭引号重复组成：原文排版瑕疵（P2.1 发现 12青金石:2540 的 」」），
      重复字符保留进 text 并告警 duplicate_closing_quote（不丢数据、不改原文）；
    - 其他非空内容：显式失败 trailing_content_after_quote，绝不静默丢弃。
    """
    stripped = after_close.strip()
    if not stripped:
        return "", []
    if set(stripped) == {close_char}:
        return stripped, [duplicate_closing_quote_warning(len(stripped))]
    raise ScriptParseError(
        source_path,
        line_no,
        TRAILING_CONTENT_AFTER_QUOTE,
        f"闭引号后存在非空内容: {stripped!r}",
    )


@dataclass
class _PendingDialogue:
    """已开启、尚未见到闭引号的台词的累积状态。"""

    speaker: str
    quote_style: QuoteStyle
    close_char: str
    start_line: int
    parts: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.parts)


class _ScriptParser:
    """单文件逐行状态机：feed 每物理行，finish 收尾，segments 收集结果。"""

    def __init__(self, source_path: str, total_lines: int):
        self._source_path = source_path
        self._total_lines = total_lines
        self._segments: list[ScriptSegment] = []
        self._narration: list[tuple[int, str]] = []
        self._pending: _PendingDialogue | None = None

    @property
    def segments(self) -> list[ScriptSegment]:
        return self._segments

    def feed(self, line_no: int, line: str) -> None:
        match = _TAG_LINE_RE.match(line)
        if match is not None:
            self._flush_narration()
            self._cut_pending_dialogue(next_speaker_line=line_no)
            self._open_dialogue(line_no, line, match)
            return
        if self._pending is not None:
            self._continue_dialogue(line_no, line)
            return
        if line.strip():
            self._narration.append((line_no, line))
        else:
            self._flush_narration()

    def finish(self) -> None:
        self._cut_pending_dialogue(next_speaker_line=None)
        self._flush_narration()

    # ---------- 台词 ----------

    def _open_dialogue(self, line_no: int, line: str, match: re.Match[str]) -> None:
        open_char = match.group(2)
        close_char, quote_style = _OPEN_QUOTES[open_char]
        rest = line[match.end(2) :]
        close_pos = rest.find(close_char)
        if close_pos >= 0:
            extra, warnings = _after_close_resolution(self._source_path, line_no, rest[close_pos + 1 :], close_char)
            # 单行台词：text 从开引号起，到闭引号止（含两侧引号）；
            # 重复闭引号瑕疵时按 _after_close_resolution 追加保留
            text = line[match.start(2) : match.end(2) + close_pos + 1] + extra
            self._emit_dialogue(line_no, line_no, match.group(1).strip(), quote_style, text, warnings)
            return
        pending = _PendingDialogue(
            speaker=match.group(1).strip(),
            quote_style=quote_style,
            close_char=close_char,
            start_line=line_no,
        )
        pending.parts.append(line[match.start(2) :])
        self._pending = pending

    def _continue_dialogue(self, line_no: int, line: str) -> None:
        pending = self._pending
        assert pending is not None
        close_pos = line.find(pending.close_char)
        if close_pos >= 0:
            extra, warnings = _after_close_resolution(
                self._source_path, line_no, line[close_pos + 1 :], pending.close_char
            )
            pending.parts.append(line[: close_pos + 1] + extra)
            self._emit_dialogue(
                pending.start_line, line_no, pending.speaker, pending.quote_style, pending.text(), warnings
            )
            self._pending = None
            return
        pending.parts.append(line)

    def _cut_pending_dialogue(self, next_speaker_line: int | None) -> None:
        """未闭合台词截断：保留已累积原文（含开引号），写告警，不补闭引号。"""
        if self._pending is None:
            return
        pending = self._pending
        end_line = self._total_lines if next_speaker_line is None else next_speaker_line - 1
        warning = unclosed_quote_warning(next_speaker_line, self._total_lines)
        self._emit_dialogue(
            pending.start_line,
            max(end_line, pending.start_line),
            pending.speaker,
            pending.quote_style,
            pending.text(),
            [warning],
        )
        self._pending = None

    def _emit_dialogue(
        self,
        line_start: int,
        line_end: int,
        speaker: str,
        quote_style: QuoteStyle,
        text: str,
        warnings: list[str],
    ) -> None:
        self._add(SegmentType.dialogue, line_start, line_end, text, speaker, quote_style, warnings)

    # ---------- 叙述 ----------

    def _flush_narration(self) -> None:
        if not self._narration:
            return
        start_line = self._narration[0][0]
        end_line = self._narration[-1][0]
        text = "\n".join(content for _, content in self._narration)
        self._narration.clear()
        self._add(SegmentType.narration, start_line, end_line, text, None, QuoteStyle.none, [])

    # ---------- 通用 ----------

    def _add(
        self,
        segment_type: SegmentType,
        line_start: int,
        line_end: int,
        text: str,
        speaker: str | None,
        quote_style: QuoteStyle,
        warnings: list[str],
    ) -> None:
        segment = ScriptSegment(
            id=_stable_segment_id(self._source_path, segment_type, line_start, line_end),
            segment_type=segment_type,
            text=text,
            source=SourceSpan(source_path=self._source_path, line_start=line_start, line_end=line_end),
            speaker=speaker,
            quote_style=quote_style,
            warnings=warnings,
        )
        self._segments.append(segment)


def parse_script_text(text: str, source_path: str) -> list[ScriptSegment]:
    """解析单个脚本文本，返回内存中的段列表（不写文件）。

    末行若为单独的 DOS EOF 标记（\\x1a），该行被忽略，并在**最后一个段**的
    warnings 追加 dos_eof_marker 告警（与 unclosed_quote / duplicate_closing_quote
    一致走段级告警通道；文件无任何段时没有可挂载目标，不产生段级告警）。
    """
    lines, has_dos_eof = _split_lines_with_eof_flag(text)
    parser = _ScriptParser(source_path, total_lines=len(lines))
    for line_no, line in enumerate(lines, start=1):
        parser.feed(line_no, line)
    parser.finish()
    if has_dos_eof and parser.segments:
        parser.segments[-1].warnings.append(dos_eof_marker_warning())
    return parser.segments


def parse_script_file(path: Path, *, source_path: str | None = None) -> list[ScriptSegment]:
    """解析单个脚本文件（UTF-8）。

    source_path 默认使用 path.name（便携，不泄露本机绝对路径）；
    需要目录级溯源时由调用方显式传入（如 gametext/纸上魔法使/xxx.txt）。
    """
    return parse_script_text(path.read_text(encoding="utf-8"), source_path or path.name)


def parse_script_directory(root: Path, *, source_prefix: str | None = None) -> list[ScriptSegment]:
    """按相对 POSIX 路径确定顺序解析 root 下全部 *.txt，拼接返回。

    - source_path 为相对 root 的 POSIX 路径（跨机器/目录可复现，不写入本机绝对路径）；
    - 提供 source_prefix 时拼接为 f"{source_prefix}/{相对路径}"，
      例如 source_prefix="gametext/纸上魔法使" 得到
      "gametext/纸上魔法使/1翡翠的排挤原理.txt"。
    """
    segments: list[ScriptSegment] = []
    files = sorted(root.glob("*.txt"), key=lambda p: p.relative_to(root).as_posix())
    for path in files:
        rel = path.relative_to(root).as_posix()
        source_path = f"{source_prefix}/{rel}" if source_prefix else rel
        segments.extend(parse_script_file(path, source_path=source_path))
    return segments
