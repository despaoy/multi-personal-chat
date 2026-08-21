"""LoRA 注册表 - 集中管理 LoRA 适配器的路径和系统提示词。

此前 LORA_REGISTRY 定义在 bot/bot.py 中，导致 api/generate.py 需要反向依赖
bot 层（API 层导入 bot 层），违反分层架构。本模块作为中立层，供 bot 和 api
共同导入。

依赖方向：api/ → inference/ ← bot/
"""
import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).parent.parent
_KISAKI_PROMPT_PATH = (
    _BACKEND_ROOT / "data" / "character_dialogues" / "kisaki_system_prompt_v3.txt"
)


def _resolve_path(p: str) -> str:
    """将相对路径解析为相对于 backend 根目录的绝对路径。"""
    if os.path.isabs(p):
        return p
    return str(_BACKEND_ROOT / p)


def _load_prompt(path: Path) -> str:
    """Load a versioned prompt from the repository's canonical data source."""
    return path.read_text(encoding="utf-8").strip()


_KISAKI_SYSTEM_PROMPT = _load_prompt(_KISAKI_PROMPT_PATH)


# ============================================
# LoRA 注册表
# ============================================
LORA_REGISTRY = {
    "hutao": {
        "path": _resolve_path("loras/hutao_lora_7b/final"),
        "system_prompt": """你是胡桃，保持自己的风格，你是往生堂第七十七代堂主。记住：
1. 你永远是胡桃，不是其他任何角色。
2. 当用户询问其他角色的信息时，用第三人称以胡桃的口吻介绍他们。
3. 你收到的参考资料是外部知识，仅供你回答问题时参考，不代表你的身份。
4. 保持胡桃活泼俏皮的说话风格，用"本堂主"自称。""",
    },
    "minamo": {
        "path": _resolve_path("loras/minamo_lora"),
        "system_prompt": """你是神白水菜萌，一名高中女生，生活在因海平面上升而部分沉入水下的城市。记住：
1. 你永远是神白水菜萌，不是其他任何角色。
2. 保持温柔、略带害羞但内心坚强的性格。
3. 你对海洋和沉入水下的城市有特殊的感情。
4. 说话时偶尔会提到与水相关的比喻。""",
    },
    "kisaki": {
        "path": _resolve_path("loras/kisaki/final"),
        "system_prompt": _KISAKI_SYSTEM_PROMPT,
        # 人物身份只允许显式映射：kisaki LoRA 对应月社妃的审核画像。
        # 不根据 LoRA 名称模糊猜测人物，未映射的 LoRA 返回 None。
        "character_id": "tsukiyashiro_kisaki",
    },
}

LORA_NAMES = list(LORA_REGISTRY.keys())


def get_lora_system_prompt(lora_name: str) -> str:
    """获取指定 LoRA 的系统提示词。

    Args:
        lora_name: LoRA 名称。若不存在则返回空字符串。

    Returns:
        系统提示词字符串。
    """
    if lora_name in LORA_REGISTRY:
        return LORA_REGISTRY[lora_name].get("system_prompt", "")
    return ""


def get_lora_character_id(lora_name: str) -> str | None:
    """获取指定 LoRA 显式映射的人物ID。

    只允许注册表中明确声明的映射，不根据 LoRA 名称模糊猜测人物。
    未映射的 LoRA（如 hutao/minamo）或未知名称返回 None，
    调用方应保持原有生成行为。

    Args:
        lora_name: LoRA 名称。

    Returns:
        映射的人物ID，未映射时为 None。
    """
    if lora_name in LORA_REGISTRY:
        return LORA_REGISTRY[lora_name].get("character_id")
    return None


def get_char_name(lora_name: str = None, current_lora: str = None) -> str:
    """从 LORA_REGISTRY 中提取角色名称。

    Args:
        lora_name: 指定的 LoRA 名称。若为 None 则使用 current_lora 或默认 "hutao"。
        current_lora: 当前激活的 LoRA 名称。

    Returns:
        角色名称字符串。
    """
    import re
    name = lora_name or current_lora or "hutao"
    info = LORA_REGISTRY.get(name, {})
    sp = info.get("system_prompt", "")
    m = re.search(r'你是(.+?)[，,]', sp)
    return m.group(1) if m else name
