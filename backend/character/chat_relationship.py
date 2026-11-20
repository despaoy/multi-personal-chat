"""User-confirmed settings for how a role relates to a conversation partner."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

RelationshipStage = Literal["stranger", "acquaintance", "familiar", "close"]


class RelationshipDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tone: Literal["restrained", "playful"] | None = None
    initiative: Literal["wait", "offer"] | None = None

    def apply_to(self, current):
        merged = {**current, **self.model_dump(exclude_unset=True)}
        return {key: value for key, value in merged.items() if value is not None}


_GUIDANCE = {
    "tone": {
        "restrained": "交流风格偏克制，保持自然，避免主动打趣或称呼过于亲昵。",
        "playful": "在情境和人物立场允许时，可以接梗或轻松打趣。严肃话题保持合适的分寸。",
    },
    "initiative": {
        "wait": (
            "当前对话里跟随用户的话题。已有足够信息时直接回应，用陈述句收尾。"
            "用户说还想继续聊时，可以顺着已有内容接一句，或等对方继续说。"
            "不主动追问或另开话题，不用“你想聊哪一段”把问题抛回去。只有无法理解或回答用户的明确请求时才澄清。"
        ),
        "offer": "当前对话里如果时机合适，可以提一个相关问题或建议接续未聊完的话题。用户要结束时自然收尾。",
    },
}


def relationship_dimension_guidance(relationship):
    choices = relationship.get("dimensions", {})
    return "".join(
        guidance.get(choices.get(key), "") for key, guidance in _GUIDANCE.items()
    )
