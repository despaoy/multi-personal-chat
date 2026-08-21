"""人物画像注册表（profile_registry）的单元测试。

验证第二步约定：
- 月社妃画像可加载且来源字段正确；
- 画像内容不含未经权威材料确认的身份（如"学生会会长"）；
- 未知人物明确失败，不自动串到其他角色；
- 画像只加载一次，get_profile 不再读磁盘；
- 配置列表转换为不可变元组；
- 空平台/适配器被 build_user_scope 拒绝；
- 编译后的上下文包含"原作核心关系"段。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from character import (
    CharacterContext,
    CharacterProfileNotFoundError,
    CharacterProfileRegistry,
    RelationshipState,
    UserScope,
    build_user_scope,
    compile_character_context,
    get_default_profile_registry,
)
from character.profile_registry import PROFILES_DIR

KISAKI_ID = "tsukiyashiro_kisaki"


@pytest.fixture()
def registry() -> CharacterProfileRegistry:
    """使用生产画像目录的独立注册表实例。"""
    reg = CharacterProfileRegistry()
    reg.load_profiles()
    return reg


def _write_profile(directory: Path, data: dict, filename: str = "test_char.json") -> None:
    (directory / filename).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


# ============================================
# 月社妃画像加载与内容
# ============================================


def test_kisaki_profile_loads(registry: CharacterProfileRegistry):
    profile = registry.get_profile(KISAKI_ID)
    assert profile.character_id == KISAKI_ID
    assert profile.display_name == "月社妃"
    assert profile.version == "v1"


def test_kisaki_profile_contains_required_sections(registry: CharacterProfileRegistry):
    profile = registry.get_profile(KISAKI_ID)
    # 画像包含人物特征、语言风格、行为边界和核心关系
    assert len(profile.traits) >= 5
    assert len(profile.speaking_style) >= 5
    assert 3 <= len(profile.boundaries) <= 4
    assert len(profile.canonical_relationships) == 4
    # 原作核心关系覆盖四位原作人物
    relationships_text = "\n".join(profile.canonical_relationships)
    for name in ("琉璃", "彼方", "夜子", "理央"):
        assert name in relationships_text
    # 身份描述明确作品来源
    assert "纸上的魔法使" in profile.identity
    assert profile.values


def test_kisaki_profile_has_no_unauthorized_identity(registry: CharacterProfileRegistry):
    """学生会会长未出现在权威画像材料中，不得进入配置。"""
    profile = registry.get_profile(KISAKI_ID)
    serialized = json.dumps(
        {
            "identity": profile.identity,
            "traits": profile.traits,
            "values": profile.values,
            "canonical_relationships": profile.canonical_relationships,
            "speaking_style": profile.speaking_style,
            "boundaries": profile.boundaries,
        },
        ensure_ascii=False,
    )
    assert "学生会会长" not in serialized


# ============================================
# 注册表行为
# ============================================


def test_unknown_character_id_raises_clear_error(registry: CharacterProfileRegistry):
    with pytest.raises(CharacterProfileNotFoundError, match="未找到人物画像"):
        registry.get_profile("不存在的人物")


def test_unknown_character_id_does_not_fall_back(tmp_path: Path):
    """未知人物必须明确失败，不能退回月社妃。"""
    reg = CharacterProfileRegistry(profiles_dir=tmp_path)
    _write_profile(
        tmp_path,
        {"character_id": KISAKI_ID, "display_name": "月社妃"},
    )
    reg.load_profiles()
    with pytest.raises(CharacterProfileNotFoundError):
        reg.get_profile("someone_else")
    # 确认没有悄悄返回其他角色
    assert reg.get_profile(KISAKI_ID).character_id == KISAKI_ID


def test_get_profile_does_not_reread_files(tmp_path: Path):
    """加载后删除画像文件，get_profile / 再次 load_profiles 仍正常。"""
    _write_profile(
        tmp_path,
        {"character_id": KISAKI_ID, "display_name": "月社妃"},
    )
    reg = CharacterProfileRegistry(profiles_dir=tmp_path)
    reg.load_profiles()

    # 删除磁盘上的配置，模拟"运行期间只加载一次"
    for json_file in tmp_path.glob("*.json"):
        json_file.unlink()

    assert reg.get_profile(KISAKI_ID).display_name == "月社妃"
    # 幂等：重复 load_profiles 不重新读磁盘（文件已删除也不报错）
    assert reg.load_profiles() == 1
    assert reg.get_profile(KISAKI_ID).display_name == "月社妃"


def test_config_lists_become_immutable_tuples(registry: CharacterProfileRegistry):
    profile = registry.get_profile(KISAKI_ID)
    for field in (
        profile.traits,
        profile.values,
        profile.canonical_relationships,
        profile.speaking_style,
        profile.boundaries,
    ):
        assert isinstance(field, tuple)


def test_list_profiles_returns_brief_info(registry: CharacterProfileRegistry):
    entries = registry.list_profiles()
    assert entries
    kisaki = next(e for e in entries if e["character_id"] == KISAKI_ID)
    assert kisaki["display_name"] == "月社妃"
    assert kisaki["version"] == "v1"
    # 简要列表只含三个字段，不泄漏完整画像
    assert set(kisaki.keys()) == {"character_id", "display_name", "version"}


def test_malformed_config_raises_clear_error(tmp_path: Path):
    (tmp_path / "broken.json").write_text("{invalid json", encoding="utf-8")
    reg = CharacterProfileRegistry(profiles_dir=tmp_path)
    with pytest.raises(ValueError, match="broken.json"):
        reg.load_profiles()


def test_missing_required_field_raises(tmp_path: Path):
    _write_profile(tmp_path, {"display_name": "缺ID的人物"})
    reg = CharacterProfileRegistry(profiles_dir=tmp_path)
    with pytest.raises(ValueError, match="character_id"):
        reg.load_profiles()


def test_duplicate_character_id_raises(tmp_path: Path):
    data = {"character_id": "dup_char", "display_name": "重复角色"}
    _write_profile(tmp_path, data, filename="a.json")
    _write_profile(tmp_path, data, filename="b.json")
    reg = CharacterProfileRegistry(profiles_dir=tmp_path)
    with pytest.raises(ValueError, match="重复"):
        reg.load_profiles()


def test_default_registry_is_cached_singleton():
    first = get_default_profile_registry()
    second = get_default_profile_registry()
    assert first is second
    profile = first.get_profile(KISAKI_ID)
    assert profile.display_name == "月社妃"


def test_profiles_dir_is_fixed_under_backend_data():
    """画像目录由程序固定计算，不接受用户输入拼接。"""
    assert PROFILES_DIR.name == "character_profiles"
    assert PROFILES_DIR.parent.name == "data"
    assert PROFILES_DIR.parent.parent.name == "backend"


# ============================================
# build_user_scope 拒绝空平台/适配器
# ============================================


def test_empty_platform_is_rejected():
    with pytest.raises(ValueError, match="平台为空"):
        build_user_scope(
            platform="  ", adapter="onebot", sender_id="10001",
            conversation_id="", conversation_type="private",
        )


def test_empty_adapter_is_rejected():
    with pytest.raises(ValueError, match="适配器为空"):
        build_user_scope(
            platform="qq", adapter="", sender_id="10001",
            conversation_id="", conversation_type="private",
        )


# ============================================
# 编译后的上下文包含"原作核心关系"段
# ============================================


def test_compiled_context_contains_canonical_relationships_section(
    registry: CharacterProfileRegistry,
):
    profile = registry.get_profile(KISAKI_ID)
    context = CharacterContext(
        profile=profile,
        user_scope=UserScope(
            platform="qq", adapter="onebot", sender_id="10001",
            conversation_id="20001", conversation_type="group",
        ),
        relationship=RelationshipState(stage="stranger"),
    )
    compiled = compile_character_context(context)

    assert "【原作核心关系】" in compiled.profile_context
    # 原作关系内容进入人物规则区
    assert "琉璃" in compiled.profile_context
    assert "理央" in compiled.profile_context
    # 段落顺序：价值倾向 → 原作核心关系 → 语言习惯
    profile = compiled.profile_context
    assert profile.index("【人物价值倾向】") < profile.index("【原作核心关系】")
    assert profile.index("【原作核心关系】") < profile.index("【人物语言习惯】")
    # 原作关系不会进入记忆参考区，也不进入动态区
    assert "琉璃" not in compiled.reference_context
    assert "琉璃" not in compiled.dynamic_context
