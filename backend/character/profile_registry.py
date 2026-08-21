"""人物画像注册表。

只负责从固定的 ``backend/data/character_profiles/`` 目录加载
JSON 画像并提供按 ``character_id`` 的查询。

不负责 LoRA 选择和 Prompt 生成；不访问数据库、不调用模型。

安全边界：
- 画像目录由本模块基于自身位置固定计算，用户输入（character_id）
  只用作字典键，绝不拼接进文件路径；
- 未知人物 ID 抛出明确的 CharacterProfileNotFoundError，
  不自动退回其他角色，避免角色串用；
- 配置格式错误时抛出明确异常，不静默使用空画像；
- 应用运行期间只加载一次（get_profile 不读磁盘）。
"""

from __future__ import annotations

import json
from pathlib import Path

from character.models import CharacterProfile

# 画像目录固定为 backend/data/character_profiles/，
# 由本文件位置推导，不接受外部传入的用户可控路径。
PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "character_profiles"

# JSON 中允许出现的字段（缺失的可选字段使用 CharacterProfile 默认值）
_REQUIRED_FIELDS = ("character_id", "display_name")
_LIST_FIELDS = (
    "traits",
    "values",
    "canonical_relationships",
    "speaking_style",
    "boundaries",
)


class CharacterProfileNotFoundError(KeyError):
    """请求的人物画像不存在。

    明确失败而不自动退回默认角色，避免角色串用。
    """


class _ProfileFormatError(ValueError):
    """画像 JSON 配置格式错误（内部转换用，聚合进加载异常信息）。"""


def _to_profile(data: dict, source: str) -> CharacterProfile:
    """把一份 JSON 字典转换为 CharacterProfile，格式错误时抛异常。"""
    if not isinstance(data, dict):
        raise _ProfileFormatError(f"{source}: 顶层必须是 JSON 对象")

    for field in _REQUIRED_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise _ProfileFormatError(f"{source}: 缺少必填字符串字段 {field!r}")

    kwargs: dict = {
        "character_id": data["character_id"].strip(),
        "display_name": data["display_name"].strip(),
    }

    identity = data.get("identity", "")
    if not isinstance(identity, str):
        raise _ProfileFormatError(f"{source}: identity 必须是字符串")
    kwargs["identity"] = identity.strip()

    version = data.get("version", "v1")
    if not isinstance(version, str) or not version.strip():
        raise _ProfileFormatError(f"{source}: version 必须是非空字符串")
    kwargs["version"] = version.strip()

    for field in _LIST_FIELDS:
        items = data.get(field, [])
        if not isinstance(items, list) or any(not isinstance(i, str) for i in items):
            raise _ProfileFormatError(f"{source}: {field} 必须是字符串列表")
        # 转换为不可变元组，去除空项
        kwargs[field] = tuple(item.strip() for item in items if item.strip())

    return CharacterProfile(**kwargs)


class CharacterProfileRegistry:
    """人物画像注册表：加载一次，之后按 ID 常数时间查询。"""

    def __init__(self, profiles_dir: Path | str | None = None) -> None:
        # profiles_dir 仅供测试注入临时目录；生产路径由 PROFILES_DIR 固定，
        # 不接受来自用户请求的路径。
        self._profiles_dir = Path(profiles_dir) if profiles_dir else PROFILES_DIR
        self._profiles: dict[str, CharacterProfile] = {}
        self._loaded = False

    def load_profiles(self) -> int:
        """从画像目录读取所有 JSON 并在内存中按 character_id 保存。

        返回加载的画像数量。重复调用时若已加载则直接返回（幂等），
        应用运行期间只加载一次，不每轮对话读取磁盘。
        """
        if self._loaded:
            return len(self._profiles)

        if not self._profiles_dir.is_dir():
            raise FileNotFoundError(
                f"人物画像目录不存在: {self._profiles_dir}，"
                "请确认 backend/data/character_profiles/ 已正确部署"
            )

        profiles: dict[str, CharacterProfile] = {}
        for path in sorted(self._profiles_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"人物画像 JSON 解析失败: {path.name}: {exc}") from exc

            try:
                profile = _to_profile(data, source=path.name)
            except _ProfileFormatError as exc:
                raise ValueError(f"人物画像配置格式错误: {exc}") from exc

            if profile.character_id in profiles:
                raise ValueError(
                    f"人物画像 character_id 重复: {profile.character_id} "
                    f"({path.name} 与已有配置冲突)"
                )
            profiles[profile.character_id] = profile

        self._profiles = profiles
        self._loaded = True
        return len(self._profiles)

    def get_profile(self, character_id: str) -> CharacterProfile:
        """按 character_id 返回人物画像，不触发磁盘读取。

        找不到时抛出 CharacterProfileNotFoundError，不自动退回其他人物。
        """
        if not self._loaded:
            self.load_profiles()

        key = character_id.strip() if isinstance(character_id, str) else character_id
        try:
            return self._profiles[key]
        except KeyError:
            available = ", ".join(sorted(self._profiles)) or "（无）"
            raise CharacterProfileNotFoundError(
                f"未找到人物画像: {character_id!r}，已加载: {available}"
            ) from None

    def list_profiles(self) -> tuple[dict[str, str], ...]:
        """返回已加载画像的简要列表（ID、显示名称、版本）。

        不返回完整画像内容，不涉及数据库。
        """
        if not self._loaded:
            self.load_profiles()
        return tuple(
            {
                "character_id": profile.character_id,
                "display_name": profile.display_name,
                "version": profile.version,
            }
            for profile in sorted(self._profiles.values(), key=lambda p: p.character_id)
        )


# 进程内默认注册表缓存：配置只加载一次，不需要 Redis 或复杂并发锁。
_default_registry: CharacterProfileRegistry | None = None


def get_default_profile_registry() -> CharacterProfileRegistry:
    """返回项目默认注册表实例（进程内单例，首次调用时加载）。"""
    global _default_registry
    if _default_registry is None:
        _default_registry = CharacterProfileRegistry()
        _default_registry.load_profiles()
    return _default_registry
