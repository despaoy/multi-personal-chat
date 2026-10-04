"""Explicit whole-memory mutation refusal, separate from hypothetical facts."""

from __future__ import annotations

import re

_DIRECT = re.compile(
    r"(?:^|[，,。！？!?；;\n])\s*(?:请)?(?:不要|别|不得|禁止|不允许|无需|不用)"
    r"(?P<actions>(?:新增|添加|替换|更新|修改|改动|更改|删除)"
    r"(?:(?:[、，,]|或|和|与|及|以及)*(?:新增|添加|替换|更新|修改|改动|更改|删除))*)"
    r"(?:我的|本人|个人|任何|现有|已保存的|长期)*(?:长期)?记忆"
    r"(?=[，,。！？!?；;\n]|$)"
)


def explicit_memory_read_only(message: str) -> bool:
    pairs = {"“": "”", "‘": "’", "「": "」", "『": "』", '"': '"'}
    stack, outside = [], []
    escaped = False
    for char in message:
        if escaped:
            outside.append(" " if stack else char)
            escaped = False
        elif char == "\\" and stack:
            escaped = True
            outside.append(" ")
        elif stack and char == stack[-1]:
            stack.pop()
            outside.append(" ")
        elif char in pairs:
            stack.append(pairs[char])
            outside.append(" ")
        else:
            outside.append(" " if stack else char)
    if stack or escaped:
        return False
    for match in _DIRECT.finditer("".join(outside)):
        actions = match["actions"]
        if any(word in actions for word in ("更新", "修改", "改动", "更改")):
            return True
        if any(word in actions for word in ("新增", "添加")) and "替换" in actions and "删除" in actions:
            return True
    return False
