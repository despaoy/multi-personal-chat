"""故事单元登记表（P3 元数据登记）。

17 个原文文件登记为 18 个故事单元：
- 16 个正篇文件各 1 单元（content_scope=main_story，卷号取文件名编号）；
- 《日后谈》按 P0 审计确认的固定边界（第 38 行空行）拆为 2 个单元：
  第 1–37 行宣传元叙事（promotional_meta）、第 38 行起追加剧本《萤色光景》（bonus_story）。

登记原则（P0.2 决议）：
- 不根据文件名预设 route / continuity_id（留给剧情结构人工审核）；
- viewpoint 一律 None：视点为文本观察值，尚未逐卷人工核实，不在登记阶段臆填；
- 同编号多文件（6/8/9 各两卷）仅共享 volume_number，连续性关系不判定。
"""

from __future__ import annotations

from dataclasses import dataclass

from knowledge.game_rag.models import ContentScope, ScriptSegment

SOURCE_PREFIX = "gametext/纸上魔法使"
EPILOGUE_SOURCE = f"{SOURCE_PREFIX}/日后谈.txt"
# P0 审计 §3.2：第 1–37 行宣传元叙事；第 38 行空行；第 39 行起追加剧本。
EPILOGUE_FIXED_BOUNDARY = 38


@dataclass(frozen=True)
class StoryUnit:
    """一个故事单元的登记元数据（对应 StoryContext 的文件级默认值）。"""

    unit_id: str
    source_path: str
    story_title: str
    volume_number: int | None
    content_scope: ContentScope
    viewpoint: str | None
    line_start: int  # 1-based，含
    line_end: int | None  # None = 到文件末行


def _main_unit(volume: int, stem: str) -> StoryUnit:
    return StoryUnit(
        unit_id=f"vol{volume:02d}_{stem}",
        source_path=f"{SOURCE_PREFIX}/{stem}.txt",
        story_title=stem,
        volume_number=volume,
        content_scope=ContentScope.main_story,
        viewpoint=None,
        line_start=1,
        line_end=None,
    )


STORY_UNITS: tuple[StoryUnit, ...] = (
    _main_unit(1, "1翡翠的排挤原理"),
    _main_unit(2, "2红宝石的天作之合"),
    _main_unit(3, "3蓝宝石的存在证明"),
    _main_unit(4, "4紫水晶的怪异传说"),
    _main_unit(5, "5磷灰石的怠惰现象"),
    _main_unit(6, "6芙蓉石的长年隔绝"),
    _main_unit(6, "6芙蓉石的终焉轮回"),
    _main_unit(7, "7黑珍珠的求爱信号"),
    _main_unit(8, "8萤石的怠惰现象"),
    _main_unit(8, "8萤石的时空残影"),
    _main_unit(9, "9白珍珠的泡沫爱慕"),
    _main_unit(9, "9绿幽灵水晶的命运连锁"),
    _main_unit(10, "10黑曜石的因果目录"),
    _main_unit(11, "11黑玛瑙的不在证明"),
    _main_unit(12, "12青金石的幻想图书馆"),
    _main_unit(13, "13璀璨的紫翠玉"),
    StoryUnit(
        unit_id="epilogue_meta",
        source_path=EPILOGUE_SOURCE,
        story_title="日后谈·宣传元叙事",
        volume_number=None,
        content_scope=ContentScope.promotional_meta,
        viewpoint=None,
        line_start=1,
        line_end=EPILOGUE_FIXED_BOUNDARY - 1,
    ),
    StoryUnit(
        unit_id="epilogue_bonus",
        source_path=EPILOGUE_SOURCE,
        story_title="日后谈·萤色光景（追加剧本）",
        volume_number=None,
        content_scope=ContentScope.bonus_story,
        viewpoint=None,
        line_start=EPILOGUE_FIXED_BOUNDARY,
        line_end=None,
    ),
)

assert len(STORY_UNITS) == 18
assert len({u.unit_id for u in STORY_UNITS}) == 18
assert len({u.source_path for u in STORY_UNITS}) == 17


def units_for_source(source_path: str) -> list[StoryUnit]:
    """返回某源文件对应的单元（仅《日后谈》为 2 个，其余 1 个）。"""
    return [u for u in STORY_UNITS if u.source_path == source_path]


def unit_by_id(unit_id: str) -> StoryUnit:
    for unit in STORY_UNITS:
        if unit.unit_id == unit_id:
            return unit
    raise KeyError(f"未知 story_unit_id: {unit_id}")


def split_segments_by_unit(segments: list[ScriptSegment]) -> dict[str, list[ScriptSegment]]:
    """把解析段按登记单元分组（保持原顺序）。

    - source_path 必须与登记表一致（提示使用 source_prefix 的便携路径）；
    - 任何段不得跨越《日后谈》固定边界（第 37/38 行之间），否则报错。
    """
    known_paths = {u.source_path for u in STORY_UNITS}
    grouped: dict[str, list[ScriptSegment]] = {u.unit_id: [] for u in STORY_UNITS}
    for seg in segments:
        path = seg.source.source_path
        if path not in known_paths:
            raise ValueError(f"segment source_path 未在登记表中: {path!r}（应为 {SOURCE_PREFIX}/<文件名>.txt）")
        placed = False
        for unit in units_for_source(path):
            end = unit.line_end if unit.line_end is not None else seg.source.line_end
            if unit.line_start <= seg.source.line_start and seg.source.line_end <= end:
                grouped[unit.unit_id].append(seg)
                placed = True
                break
        if not placed:
            raise ValueError(f"segment 跨越固定单元边界: {path} L{seg.source.line_start}-{seg.source.line_end}")
    return grouped
