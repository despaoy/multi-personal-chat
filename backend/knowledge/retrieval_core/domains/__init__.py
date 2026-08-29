"""内置知识域配置包。

每个作品/知识域一个模块，只包含该域专属的：
- 数据源路径
- 别名表（alias → 规范名）
- 卷名/故事标题表
- 叙事层取值与权重策略

通用检索代码（query/retrieval/pipeline）不 import 本包任何具体
模块；注册表通过 builtin_domain_factories() 惰性发现它们。
新增域 = 新增一个工厂函数并注册到 _FACTORIES。
"""

from __future__ import annotations

from collections.abc import Callable

from ..registry import KnowledgeDomainConfig

DomainFactory = Callable[[], KnowledgeDomainConfig]


def tsukiyashiro_kisaki_domain() -> KnowledgeDomainConfig:
    """纸上魔法使（月社妃）知识域：首个正式数据域。

    数据源为 P5 approved 知识卡（fact/relation/event），
    叙事层取值沿用 P4 元数据契约。
    """
    from pathlib import Path

    from ..loaders import AliasEntityNormalizer, ApprovedCardsLoader
    from ..registry import NarrativeLayerPolicy

    backend_root = Path(__file__).resolve().parents[3]
    source_root = backend_root / "data" / "knowledge" / "tsukiyashiro_kisaki" / "knowledge_candidate_review"

    aliases = {
        # 规范名（自映射，保证单字名可命中）
        "夜子": "夜子",
        "琉璃": "琉璃",
        "妃": "妃",
        "理央": "理央",
        "汀": "汀",
        "岬": "岬",
        "彼方": "彼方",
        "暗子": "暗子",
        "萤": "萤",
        "奏": "奏",
        "克丽索贝莉露": "克丽索贝莉露",
        "魔法之书": "魔法之书",
        # 全名/别名 → 规范名
        "遊行寺夜子": "夜子",
        "游行寺夜子": "夜子",
        "新来琉璃": "琉璃",
        "月社妃": "妃",
        "小妃": "妃",
        "伏见理央": "理央",
        "遊行寺汀": "汀",
        "游行寺汀": "汀",
        "遊行寺暗子": "暗子",
        "游行寺暗子": "暗子",
        "本城岬": "岬",
        "日向彼方": "彼方",
        "克丽丝": "克丽索贝莉露",
        "魔法书": "魔法之书",
        "幻想图书馆": "幻想图书馆",
        "纸上魔法使": "纸上魔法使",
    }

    story_titles = [
        "翡翠的排挤原理",
        "红宝石的天作之合",
        "蓝宝石的存在证明",
        "紫水晶的怪异传说",
        "磷灰石的怠惰现象",
        "芙蓉石的长年隔绝",
        "芙蓉石的终焉轮回",
        "黑珍珠的求爱信号",
        "萤石的怠惰现象",
        "萤石的时空残影",
        "白珍珠的泡沫爱慕",
        "绿幽灵水晶的命运连锁",
        "黑曜石的因果目录",
        "黑玛瑙的不在证明",
        "缟玛瑙的不在证明",
        "青金石的幻想图书馆",
        "璀璨的紫翠玉",
        "日后谈·萤色光景",
        "萤色光景",
    ]

    normalizer = AliasEntityNormalizer(aliases)
    loader = ApprovedCardsLoader(
        domain_id="tsukiyashiro_kisaki",
        index_version="v1",
        entity_normalizer=normalizer,
    )

    return KnowledgeDomainConfig(
        domain_id="tsukiyashiro_kisaki",
        source_root=source_root,
        loader=loader,
        document_types=["fact", "relation", "event"],
        aliases=aliases,
        story_titles=story_titles,
        narrative_policy=NarrativeLayerPolicy(
            reality_boost={
                "objective": 1.0,
                "inferred": 1.0,
                "character_claim": 0.9,
                "fictional": 0.9,
            },
            temporal_boost={
                "current": 1.0,
                "flashback": 0.95,
                "reconstruction": 0.9,
                "hypothetical": 0.85,
            },
            scope_boost={
                "main_story": 1.0,
                "bonus_story": 0.9,
            },
        ),
        index_version="v1",
        enabled=True,
        display_name="纸上魔法使",
        description="P5 approved 知识卡（事实/关系/事件）索引域",
    )


# 域工厂注册表：新增域在此追加
_FACTORIES: list[DomainFactory] = [
    tsukiyashiro_kisaki_domain,
]


def builtin_domain_factories() -> list[DomainFactory]:
    return list(_FACTORIES)
