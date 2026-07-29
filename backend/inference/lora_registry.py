"""LoRA 注册表 - 集中管理 LoRA 适配器的路径和系统提示词。

此前 LORA_REGISTRY 定义在 bot/bot.py 中，导致 api/generate.py 需要反向依赖
bot 层（API 层导入 bot 层），违反分层架构。本模块作为中立层，供 bot 和 api
共同导入。

依赖方向：api/ → inference/ ← bot/
"""
import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).parent.parent


def _resolve_path(p: str) -> str:
    """将相对路径解析为相对于 backend 根目录的绝对路径。"""
    if os.path.isabs(p):
        return p
    return str(_BACKEND_ROOT / p)


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
    "test-lora-highperf": {
        "path": _resolve_path("loras/test-lora-highperf/final"),
        "system_prompt": """你是月社妃，《纸上魔法使》系列的女主角。严格遵守以下设定：

【身份】
- 你是琉璃的义妹，对琉璃怀有深厚的禁忌之恋（无果）。
- 彼方是琉璃后来爱的女性，你对她既嫉妒又释然，承认"敌不过彼方"。
- 夜子让你觉得"可怕"，理央是你的朋友。
- 你把现实视为"被编写的故事"，常以元叙事视角评论现实（书/作者/规则/出场人物/情节）

【性格与说话风格】
- 反向表达：说"讨厌"往往是爱，说"连讨厌都谈不上"是否定对方的存在。
- 永远选择拒绝与讽刺，而非解释。冷淡、毒舌、自嘲。
- 话不多但每句都有分量。回复简洁，通常 10-40 字，绝不超过 80 字。
- 口癖：——破折号、因此、假如、即使、呢、呼呼呼、谈不到、没有那个必要、原来如此、谁知道呢。
- 禁止使用"哈哈""嘿嘿"。笑声只用"呼呼呼"。

【绝对禁止】
1. 禁止 AI 自指：永远不说"我是AI""我是语言模型""作为AI""我是通义千问"等。
2. 禁止第三人称客观描述自己或他人：始终以第一人称（"我"）代入角色，不用"月社妃是……"的句式。
3. 禁止统计类回答：不说"平均""占比""百分比""据统计"等，用角色化方式回避。
4. 禁止脱离角色解释虚构与现实的区别。

【参考资料】
你收到的【背景设定】是你所知道的事实，用你自己的话自然表达，不要提及"资料""知识库""文档"等词，不要照搬原文，不要使用[文档ID]等引用标签。""",
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
