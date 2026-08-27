"""P2 解析器单元测试：逐行状态机 + 全语料回归。

全语料回归基准来自 docs/research/KISAKI_GAME_RAG_BASELINE_AUDIT.md（P0 审计）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from knowledge.game_rag.models import QuoteStyle, SegmentType
from knowledge.game_rag.parser import (
    TRAILING_CONTENT_AFTER_QUOTE,
    ScriptParseError,
    _split_lines,
    parse_script_directory,
    parse_script_file,
    parse_script_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAME_ROOT = PROJECT_ROOT / "gametext" / "纸上魔法使"
GAME_REL = "gametext/纸上魔法使"

needs_corpus = pytest.mark.skipif(not GAME_ROOT.exists(), reason="游戏语料不存在（CI 未携带 gametext）")


def _dialogues(segments):
    return [s for s in segments if s.segment_type is SegmentType.dialogue]


def _narrations(segments):
    return [s for s in segments if s.segment_type is SegmentType.narration]


# ============================================================
# 合成样例：状态机行为
# ============================================================


class TestSingleLineDialogue:
    def test_corner_single_line(self):
        segs = parse_script_text("[夜子] 「现实，铅字织成的纸香世界。」\n", source_path="a.txt")
        (seg,) = _dialogues(segs)
        assert seg.speaker == "夜子"
        assert seg.text == "「现实，铅字织成的纸香世界。」"
        assert seg.quote_style is QuoteStyle.corner
        assert (seg.source.line_start, seg.source.line_end) == (1, 1)
        assert seg.warnings == []

    def test_curly_single_line(self):
        segs = parse_script_text("[妃] “喵。”\n", source_path="a.txt")
        (seg,) = _dialogues(segs)
        assert seg.text == "“喵。”"
        assert seg.quote_style is QuoteStyle.curly

    def test_double_corner_single_line(self):
        segs = parse_script_text("[妃] 『内层独白。』\n", source_path="a.txt")
        (seg,) = _dialogues(segs)
        assert seg.text == "『内层独白。』"
        assert seg.quote_style is QuoteStyle.double_corner

    def test_speaker_label_stripped(self):
        segs = parse_script_text("[ 妃 ] 「喵。」\n", source_path="a.txt")
        (seg,) = _dialogues(segs)
        assert seg.speaker == "妃"

    def test_text_excludes_speaker_tag_and_keeps_quotes(self):
        segs = parse_script_text("[妃] 「你好。」\n", source_path="a.txt")
        (seg,) = _dialogues(segs)
        assert not seg.text.startswith("[")
        assert seg.text.startswith("「") and seg.text.endswith("」")


class TestMultilineDialogue:
    def test_legitimate_multiline(self):
        text = "[妃] 「我讨厌大海，\n受不了海风吹乱头发。」\n"
        (seg,) = _dialogues(parse_script_text(text, source_path="a.txt"))
        assert (seg.source.line_start, seg.source.line_end) == (1, 2)
        assert seg.text == "「我讨厌大海，\n受不了海风吹乱头发。」"
        assert seg.warnings == []

    def test_multiline_quote_style_preserved(self):
        text = "[妃] \u201c我发自心底\n喜欢琉璃。\u201d\n"
        (seg,) = _dialogues(parse_script_text(text, source_path="a.txt"))
        assert seg.quote_style is QuoteStyle.curly
        assert seg.text == "\u201c我发自心底\n喜欢琉璃。\u201d"


class TestUnclosedDialogue:
    def test_unclosed_cut_before_next_speaker(self):
        text = "[夜子] 「未闭合\n中间叙述\n[琉璃] 「新台词。」\n"
        segs = parse_script_text(text, source_path="a.txt")
        dialogues = _dialogues(segs)
        assert len(dialogues) == 2
        first, second = dialogues
        # 前一条：截断为 dialogue，保留开引号，带告警
        assert first.speaker == "夜子"
        assert (first.source.line_start, first.source.line_end) == (1, 2)
        assert first.text == "「未闭合\n中间叙述"
        assert first.warnings == ["unclosed_quote: next_speaker_line=3"]
        # 后一条：未被吞并，独立完整解析
        assert second.speaker == "琉璃"
        assert second.text == "「新台词。」"
        assert second.warnings == []
        # 中间叙述行进入未闭合台词 span，不产生独立 narration
        assert _narrations(segs) == []

    def test_unclosed_until_eof(self):
        text = "[妃] 「未闭合\n叙述到文件尾\n"
        (seg,) = _dialogues(parse_script_text(text, source_path="a.txt"))
        assert (seg.source.line_start, seg.source.line_end) == (1, 2)
        assert seg.text == "「未闭合\n叙述到文件尾"
        assert seg.warnings == ["unclosed_quote: next_speaker_line=EOF(file_end_line=2)"]

    def test_unclosed_does_not_swallow_following_speaker(self):
        """7 处吞并问题的最小复现：被吞位置的台词必须独立成段。"""
        text = "[理央] 「呼呼呼，敬请期待！\n跟平常一样。\n[琉璃] 「为自己做早餐的女孩子。」\n"
        segs = parse_script_text(text, source_path="a.txt")
        dialogues = _dialogues(segs)
        assert [(d.speaker, d.source.line_start, d.source.line_end) for d in dialogues] == [
            ("理央", 1, 2),
            ("琉璃", 3, 3),
        ]
        assert dialogues[1].text == "「为自己做早餐的女孩子。」"


class TestNarration:
    def test_consecutive_narration_merged(self):
        text = "第一行叙述。\n第二行叙述。\n[妃] 「台词。」\n第三行叙述。\n"
        segs = parse_script_text(text, source_path="a.txt")
        narrations = _narrations(segs)
        assert len(narrations) == 2
        first, second = narrations
        assert first.text == "第一行叙述。\n第二行叙述。"
        assert (first.source.line_start, first.source.line_end) == (1, 2)
        assert (second.source.line_start, second.source.line_end) == (4, 4)

    def test_narration_field_constraints(self):
        segs = parse_script_text("她的名字叫月社妃。\n", source_path="a.txt")
        (seg,) = _narrations(segs)
        assert seg.speaker is None
        assert seg.quote_style is QuoteStyle.none

    def test_blank_lines_do_not_create_empty_narration(self):
        text = "叙述A\n\n\n叙述B\n"
        segs = parse_script_text(text, source_path="a.txt")
        narrations = _narrations(segs)
        assert [n.text for n in narrations] == ["叙述A", "叙述B"]
        # 空白间隔可由相邻 SourceSpan 推断：1 与 4 之间隔 2 行
        assert (narrations[0].source.line_end, narrations[1].source.line_start) == (1, 4)

    def test_blank_line_ends_narration_block(self):
        text = "叙述A\n \n叙述B\n"
        narrations = _narrations(parse_script_text(text, source_path="a.txt"))
        assert len(narrations) == 2

    def test_no_narration_for_pure_blank_text(self):
        assert parse_script_text("\n\n  \n", source_path="a.txt") == []


class TestStableIds:
    def test_same_input_same_ids(self):
        text = "[妃] 「台词。」\n叙述。\n"
        first = parse_script_text(text, source_path="a.txt")
        second = parse_script_text(text, source_path="a.txt")
        assert [s.id for s in first] == [s.id for s in second]
        assert [s.model_dump() for s in first] == [s.model_dump() for s in second]

    def test_different_source_path_different_ids(self):
        text = "[妃] 「台词。」\n"
        a = parse_script_text(text, source_path="a.txt")
        b = parse_script_text(text, source_path="b.txt")
        assert a[0].id != b[0].id

    def test_id_independent_of_preceding_segmentation(self):
        """ID 不含段序号：同一位置的段不因前面无关段落切分变化而改变 ID。"""
        # 文本 A：第 1、3 行叙述被空行切成两块 → 台词是第 3 个段
        text_a = "叙述一。\n\n叙述二。\n[妃] 「台词。」\n"
        # 文本 B：空行在第 1 行，第 2-3 行叙述合并成一块 → 台词是第 2 个段
        text_b = "\n叙述一。\n叙述二。\n[妃] 「台词。」\n"
        segs_a = parse_script_text(text_a, source_path="a.txt")
        segs_b = parse_script_text(text_b, source_path="a.txt")
        assert len(segs_a) == 3 and len(segs_b) == 2  # 前面切分确实不同
        da = [s for s in segs_a if s.segment_type is SegmentType.dialogue][0]
        db = [s for s in segs_b if s.segment_type is SegmentType.dialogue][0]
        assert (da.source.line_start, da.source.line_end) == (db.source.line_start, db.source.line_end) == (4, 4)
        assert da.id == db.id


class TestFileAndDirectoryApi:
    def test_parse_file_default_source_path_uses_portable_name(self, tmp_path):
        """默认 source_path 为 path.name，不得泄露本机绝对路径。"""
        path = tmp_path / "scene.txt"
        path.write_text("[妃] 「喵。」\n", encoding="utf-8")
        (seg,) = _dialogues(parse_script_file(path))
        assert seg.source.source_path == "scene.txt"
        assert str(tmp_path) not in seg.source.source_path

    def test_parse_file_explicit_source_path(self, tmp_path):
        path = tmp_path / "scene.txt"
        path.write_text("[妃] 「喵。」\n", encoding="utf-8")
        (seg,) = _dialogues(parse_script_file(path, source_path="gametext/x.txt"))
        assert seg.source.source_path == "gametext/x.txt"

    def test_parse_file_crlf_handled(self, tmp_path):
        path = tmp_path / "crlf.txt"
        path.write_bytes("[妃] 「我讨厌大海，\r\n受不了海风吹乱头发。」\r\n".encode())
        (seg,) = _dialogues(parse_script_file(path, source_path="crlf.txt"))
        assert seg.text == "「我讨厌大海，\n受不了海风吹乱头发。」"
        assert (seg.source.line_start, seg.source.line_end) == (1, 2)

    def test_directory_parses_in_deterministic_order(self, tmp_path):
        (tmp_path / "b.txt").write_text("[妃] 「B。」\n", encoding="utf-8")
        (tmp_path / "a.txt").write_text("[妃] 「A。」\n", encoding="utf-8")
        (tmp_path / "c.md").write_text("[妃] 「忽略。」\n", encoding="utf-8")
        segs = parse_script_directory(tmp_path)
        assert [s.text for s in _dialogues(segs)] == ["「A。」", "「B。」"]


class TestPortableProvenance:
    """P2.1：溯源与 ID 不依赖机器绝对路径，跨目录/机器可复现。"""

    def test_same_file_in_two_dirs_same_source_path_and_ids(self, tmp_path):
        text = "[妃] 「喵。」\n叙述。\n"
        dir_a = tmp_path / "corpus-a"
        dir_b = tmp_path / "deep" / "corpus-b"
        dir_a.mkdir()
        dir_b.mkdir(parents=True)
        (dir_a / "scene.txt").write_text(text, encoding="utf-8")
        (dir_b / "scene.txt").write_text(text, encoding="utf-8")
        segs_a = parse_script_directory(dir_a)
        segs_b = parse_script_directory(dir_b)
        assert [s.source.source_path for s in segs_a] == ["scene.txt", "scene.txt"]
        assert [(s.id, s.source.source_path) for s in segs_a] == [(s.id, s.source.source_path) for s in segs_b]
        assert not any(str(tmp_path) in s.source.source_path for s in segs_a + segs_b)

    def test_directory_source_prefix(self, tmp_path):
        (tmp_path / "1翡翠的排挤原理.txt").write_text("[妃] 「喵。」\n", encoding="utf-8")
        (seg,) = _dialogues(parse_script_directory(tmp_path, source_prefix="gametext/纸上魔法使"))
        assert seg.source.source_path == "gametext/纸上魔法使/1翡翠的排挤原理.txt"

    def test_directory_sorts_by_relative_posix_path(self, tmp_path):
        (tmp_path / "10黑曜石.txt").write_text("[妃] 「十。」\n", encoding="utf-8")
        (tmp_path / "1翡翠.txt").write_text("[妃] 「一。」\n", encoding="utf-8")
        (tmp_path / "2红宝石.txt").write_text("[妃] 「二。」\n", encoding="utf-8")
        segs = parse_script_directory(tmp_path)
        # POSIX 字典序："10黑曜石" < "1翡翠"（'0' < '翡'）< "2红宝石"
        assert [s.text for s in _dialogues(segs)] == ["「十。」", "「一。」", "「二。」"]


class TestTrailingContentAfterQuote:
    """P2.1：闭引号后不得静默丢弃内容。"""

    def test_single_line_trailing_content_rejected(self):
        with pytest.raises(ScriptParseError) as excinfo:
            parse_script_text("[妃] 「台词。」尾部叙述\n", source_path="a.txt")
        err = excinfo.value
        assert err.code == TRAILING_CONTENT_AFTER_QUOTE
        assert err.source_path == "a.txt"
        assert err.line_no == 1

    def test_multiline_trailing_content_rejected(self):
        text = "[妃] 「跨行\n台词。」尾部\n"
        with pytest.raises(ScriptParseError) as excinfo:
            parse_script_text(text, source_path="a.txt")
        err = excinfo.value
        assert err.code == TRAILING_CONTENT_AFTER_QUOTE
        assert err.line_no == 2  # 闭引号所在物理行

    def test_trailing_whitespace_allowed(self):
        (seg,) = _dialogues(parse_script_text("[妃] 「台词。」  \n", source_path="a.txt"))
        assert seg.text == "「台词。」"
        assert seg.warnings == []

    def test_duplicate_closing_quote_tolerated_with_warning(self):
        """原文排版瑕疵（重复闭引号，如 12青金石:2540 的 」」）：保留原文 + 稳定告警。"""
        (seg,) = _dialogues(parse_script_text("[妃] 「不要啊啊！」」\n", source_path="a.txt"))
        assert seg.text == "「不要啊啊！」」"
        assert seg.warnings == ["duplicate_closing_quote: count=1"]


# ============================================================
# 全语料回归（基准：P0 审计）
# ============================================================

# 7 处已知吞没位置：截断点（吞没者起点）与被吞台词（现在独立解析）
EXPECTED_UNCLOSED = [
    # (文件, 吞没者行, 说话人, 截止行, 下一条标签行)
    ("12青金石的幻想图书馆.txt", 2485, "夜子", 2487, 2488),
    ("1翡翠的排挤原理.txt", 3892, "岬", 3893, 3894),
    ("2红宝石的天作之合.txt", 70, "暗子", 71, 72),
    ("4紫水晶的怪异传说.txt", 326, "彼方", 327, 328),
    ("6芙蓉石的终焉轮回.txt", 228, "理央", 229, 230),
    ("9白珍珠的泡沫爱慕.txt", 1641, "琉璃", 1644, 1645),
    ("日后谈.txt", 183, "彼方", 185, 186),
]

# 7 条此前被吞、现在必须独立解析的台词
EXPECTED_RECOVERED = [
    ("12青金石的幻想图书馆.txt", 2488, "琉璃"),
    ("1翡翠的排挤原理.txt", 3894, "岬"),
    ("2红宝石的天作之合.txt", 72, "夜子"),
    ("4紫水晶的怪异传说.txt", 328, "彼方"),
    ("6芙蓉石的终焉轮回.txt", 230, "琉璃"),
    ("9白珍珠的泡沫爱慕.txt", 1645, "琉璃"),
    ("日后谈.txt", 186, "彼方"),
]

# 9 条合法跨行台词（P0 审计 §2.2 影响评估）
EXPECTED_MULTILINE = [
    ("10黑曜石的因果目录.txt", 836, 837, "琉璃"),
    ("1翡翠的排挤原理.txt", 46, 47, "妃"),
    ("1翡翠的排挤原理.txt", 59, 60, "妃"),
    ("1翡翠的排挤原理.txt", 148, 149, "暗子"),
    ("1翡翠的排挤原理.txt", 378, 379, "汀"),
    ("1翡翠的排挤原理.txt", 1660, 1661, "夜子"),
    ("3蓝宝石的存在证明.txt", 555, 556, "琉璃"),
    ("3蓝宝石的存在证明.txt", 2536, 2537, "妃"),
    ("6芙蓉石的长年隔绝.txt", 491, 492, "彼方"),
]


@pytest.fixture(scope="module")
def corpus_segments():
    """全语料经目录 API（便携前缀）解析；任何文件触发硬错误都会使 fixture 失败。"""
    return parse_script_directory(GAME_ROOT, source_prefix=GAME_REL)


def _coverage_violations(lines: list[str], spans: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """覆盖检查辅助：返回（未覆盖的非空物理行, 覆盖次数>1 的物理行）。

    空白行未被覆盖属正常（解析器不产出空段），不计入违规。
    """
    counts: dict[int, int] = {}
    for start, end in spans:
        for line_no in range(start, end + 1):
            counts[line_no] = counts.get(line_no, 0) + 1
    nonblank = {i for i, text in enumerate(lines, 1) if text.strip()}
    uncovered = sorted(nonblank - counts.keys())
    overcovered = sorted(line_no for line_no, count in counts.items() if count > 1)
    return uncovered, overcovered


class TestCoverageHelper:
    """合成反例：证明覆盖检查辅助能识别缺失/重复覆盖。"""

    def test_detects_missing_nonblank_line(self):
        lines = ["叙述。", "[妃] 「喵。」", "被遗漏的叙述。"]
        spans = [(1, 1), (2, 2)]  # 第 3 行非空但未被任何段覆盖
        uncovered, overcovered = _coverage_violations(lines, spans)
        assert uncovered == [3]
        assert overcovered == []

    def test_detects_overcovered_line(self):
        lines = ["叙述。"]
        uncovered, overcovered = _coverage_violations(lines, [(1, 1), (1, 1)])
        assert uncovered == []
        assert overcovered == [1]

    def test_blank_uncovered_not_flagged(self):
        lines = ["叙述。", ""]
        uncovered, overcovered = _coverage_violations(lines, [(1, 1)])
        assert uncovered == []
        assert overcovered == []


@needs_corpus
class TestCorpusRegression:
    def test_dialogue_count(self, corpus_segments):
        assert len(_dialogues(corpus_segments)) == 17530

    def test_speaker_count(self, corpus_segments):
        assert len({s.speaker for s in _dialogues(corpus_segments)}) == 14

    def test_kisaki_dialogue_count(self, corpus_segments):
        assert sum(1 for s in _dialogues(corpus_segments) if s.speaker == "妃") == 1598

    def test_quote_style_distribution(self, corpus_segments):
        styles = [s.quote_style for s in _dialogues(corpus_segments)]
        assert sum(1 for x in styles if x is QuoteStyle.curly) == 8
        assert sum(1 for x in styles if x is QuoteStyle.double_corner) == 0

    def test_unclosed_quote_exactly_seven(self, corpus_segments):
        unclosed = [s for s in corpus_segments if any("unclosed_quote" in w for w in s.warnings)]
        assert len(unclosed) == 7

    def test_unclosed_positions_and_warnings(self, corpus_segments):
        by_start = {
            (s.source.source_path.rsplit("/", 1)[-1], s.source.line_start): s for s in _dialogues(corpus_segments)
        }
        for filename, start_line, speaker, end_line, next_line in EXPECTED_UNCLOSED:
            seg = by_start[(filename, start_line)]
            assert seg.speaker == speaker
            assert seg.source.line_end == end_line
            assert seg.warnings == [f"unclosed_quote: next_speaker_line={next_line}"]

    def test_swallowed_lines_now_parse_independently(self, corpus_segments):
        by_start = {
            (s.source.source_path.rsplit("/", 1)[-1], s.source.line_start): s for s in _dialogues(corpus_segments)
        }
        for filename, line_no, speaker in EXPECTED_RECOVERED:
            seg = by_start[(filename, line_no)]
            assert seg.speaker == speaker
            assert (seg.source.line_start, seg.source.line_end) == (line_no, line_no)
            assert seg.warnings == []
            assert seg.text.startswith("「") and seg.text.endswith("」")

    def test_legitimate_multiline_no_warning(self, corpus_segments):
        by_start = {
            (s.source.source_path.rsplit("/", 1)[-1], s.source.line_start): s for s in _dialogues(corpus_segments)
        }
        for filename, start_line, end_line, speaker in EXPECTED_MULTILINE:
            seg = by_start[(filename, start_line)]
            assert seg.speaker == speaker
            assert (seg.source.line_start, seg.source.line_end) == (start_line, end_line)
            assert seg.warnings == []
            assert "\n" in seg.text
            assert seg.text.startswith("「") and seg.text.endswith("」")

    def test_kisaki_multiline_specifically(self, corpus_segments):
        """指令要求的妃跨行台词单独断言。"""
        by_start = {
            (s.source.source_path.rsplit("/", 1)[-1], s.source.line_start): s for s in _dialogues(corpus_segments)
        }
        kisaki_cases = [c for c in EXPECTED_MULTILINE if c[3] == "妃"]
        assert len(kisaki_cases) == 3
        seg_46 = by_start[("1翡翠的排挤原理.txt", 46)]
        assert seg_46.text == "「我讨厌大海，\n受不了海风吹乱头发。」"
        seg_59 = by_start[("1翡翠的排挤原理.txt", 59)]
        assert seg_59.text == "「不用你说我也知道。\n我的头发迷倒了琉璃。」"
        seg_2536 = by_start[("3蓝宝石的存在证明.txt", 2536)]
        assert seg_2536.text == "「只要我不存在，一切就会顺利。\n位于不和中心的，一直都是我呢。」"

    def test_no_dialogue_embeds_next_speaker_tag(self, corpus_segments):
        """所有 dialogue 文本不得内嵌下一条行首说话人标签（吞并回归守卫）。"""
        nested = re.compile(r"\n\s*\[[^\[\]\n]+\]\s*[「『“]")
        offenders = [s for s in _dialogues(corpus_segments) if nested.search(s.text)]
        assert offenders == []

    def test_output_order_stable_by_file_and_line(self, corpus_segments):
        expected_files = sorted(p.name for p in GAME_ROOT.glob("*.txt"))
        order = [s.source.source_path for s in corpus_segments]
        assert order == sorted(order)
        seen_files = [f for i, f in enumerate(order) if i == 0 or order[i - 1] != f]
        assert [f.rsplit("/", 1)[-1] for f in seen_files] == expected_files

    def test_spans_valid_ordered_and_non_overlapping(self, corpus_segments):
        prev_end = None
        prev_file = None
        for seg in corpus_segments:
            assert seg.source.line_start <= seg.source.line_end
            if seg.source.source_path != prev_file:
                prev_file, prev_end = seg.source.source_path, seg.source.line_end
                continue
            assert prev_end < seg.source.line_start, (
                f"span 倒序/重叠: {prev_file} prev_end={prev_end} start={seg.source.line_start}"
            )
            prev_end = seg.source.line_end

    def test_rerun_identical(self, corpus_segments):
        second = parse_script_directory(GAME_ROOT, source_prefix=GAME_REL)
        assert [s.model_dump() for s in second] == [s.model_dump() for s in corpus_segments]

    def test_directory_prefix_equals_per_file_parsing(self, corpus_segments):
        """目录 API（source_prefix）与逐文件显式 source_path 产出完全一致的溯源与 ID。"""
        per_file = []
        for path in sorted(GAME_ROOT.glob("*.txt")):
            per_file.extend(parse_script_file(path, source_path=f"{GAME_REL}/{path.name}"))
        assert [(s.id, s.source.source_path) for s in per_file] == [
            (s.id, s.source.source_path) for s in corpus_segments
        ]

    def test_no_trailing_content_error_and_one_duplicate_quote(self, corpus_segments):
        """全语料不触发 trailing_content_after_quote（fixture 成功即证明）；
        唯一原文排版瑕疵为 12青金石:2540 的重复闭引号，保留原文并带告警。"""
        dup = [s for s in corpus_segments if any(w.startswith("duplicate_closing_quote") for w in s.warnings)]
        assert len(dup) == 1
        seg = dup[0]
        assert seg.source.source_path == f"{GAME_REL}/12青金石的幻想图书馆.txt"
        assert (seg.source.line_start, seg.source.line_end) == (2540, 2540)
        assert seg.text.endswith("」」")
        assert seg.warnings == ["duplicate_closing_quote: count=1"]

    def test_every_nonblank_line_covered_exactly_once(self, corpus_segments):
        """P2.1 覆盖检查：非空物理行恰好被一个段覆盖；未覆盖行只能是空行（32 个）。"""
        per_file_spans: dict[str, list[tuple[int, int]]] = {}
        for seg in corpus_segments:
            per_file_spans.setdefault(seg.source.source_path, []).append((seg.source.line_start, seg.source.line_end))
        assert len(per_file_spans) == 17
        total_uncovered_blank = 0
        for rel_path, spans in per_file_spans.items():
            lines = _split_lines((PROJECT_ROOT / rel_path).read_text(encoding="utf-8"))
            uncovered, overcovered = _coverage_violations(lines, spans)
            assert uncovered == [], f"{rel_path} 存在未覆盖的非空物理行: {uncovered[:10]}"
            assert overcovered == [], f"{rel_path} 存在覆盖次数>1 的物理行: {overcovered[:10]}"
            blank = {i for i, text in enumerate(lines, 1) if not text.strip()}
            covered: set[int] = set()
            for start, end in spans:
                covered.update(range(start, end + 1))
            total_uncovered_blank += len(blank - covered)
        assert total_uncovered_blank == 32  # 与 P0 审计的 32 个空行一致


# ============================================================
# DOS EOF 标记（0x1A / SUB）：仅末行严格单独出现时忽略并告警
# ============================================================


class TestDosEofMarker:
    def test_trailing_standalone_sub_ignored_with_warning(self):
        """末行单独 \x1a：不生成段、不进任何 text，告警挂最后一个段。"""
        segs = parse_script_text("[妃] 「喵。」\n尾声。\n\x1a", source_path="a.txt")
        assert [(s.source.line_start, s.source.line_end) for s in segs] == [(1, 1), (2, 2)]
        assert segs[-1].warnings == ["dos_eof_marker: trailing standalone 0x1A ignored"]
        assert all("\x1a" not in s.text for s in segs)

    def test_trailing_sub_after_final_newline_also_ignored(self):
        """ "...\n\x1a\n"：哨兵空串先移除，末行 \x1a 再移除。"""
        segs = parse_script_text("[妃] 「喵。」\n\x1a\n", source_path="a.txt")
        assert len(segs) == 1
        assert segs[-1].warnings == ["dos_eof_marker: trailing standalone 0x1A ignored"]

    def test_sub_with_other_content_not_ignored(self):
        """行内带其他内容的 \x1a：不是严格单独的 EOF 标记，按原文保留且不告警。"""
        segs = parse_script_text("尾声。\x1a", source_path="a.txt")
        (seg,) = segs
        assert seg.text == "尾声。\x1a"
        assert seg.warnings == []

    def test_sub_in_middle_line_preserved(self):
        """非末行出现 \x1a：一律按原文保留，不触发告警。"""
        segs = parse_script_text("前\x1a行\n[妃] 「喵。」\n", source_path="a.txt")
        assert segs[0].text == "前\x1a行"
        assert all("dos_eof_marker" not in w for s in segs for w in s.warnings)

    def test_only_sub_yields_no_segments(self):
        """文件仅含 \x1a：无段可挂告警，返回空列表而非崩溃。"""
        assert parse_script_text("\x1a", source_path="a.txt") == []

    @needs_corpus
    def test_corpus_dos_eof_only_in_bonus_tail(self, corpus_segments):
        """全语料唯一 DOS EOF 标记在 日后谈.txt 末行（L879）；告警挂该文件最后一段。"""
        flagged = [s for s in corpus_segments if any(w.startswith("dos_eof_marker") for w in s.warnings)]
        assert len(flagged) == 1
        seg = flagged[0]
        assert seg.source.source_path == f"{GAME_REL}/日后谈.txt"
        assert seg.source.line_end == 877  # L878 空行不入段，末段止于 877
        # 没有任何段把 \x1a 写进 text（否则会污染 scenes.jsonl）
        assert all("\x1a" not in s.text for s in corpus_segments)
