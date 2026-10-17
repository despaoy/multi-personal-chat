"""One numeric contract for partial settings updates and complete generation reads."""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerationOptions(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, hide_input_in_errors=True)

    temperature: float = Field(0.7, ge=0)
    maxTokens: int = Field(2048, gt=0)
    topP: float = Field(0.9, gt=0, le=1)

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("Generation parameters must be numeric, not boolean")
        return value


def read_generation_options(config: dict[str, Any]) -> dict[str, float | int]:
    values = {}
    for name, env_name in {
        "temperature": "VLLM_TEMPERATURE",
        "maxTokens": "VLLM_MAX_TOKENS",
        "topP": "VLLM_TOP_P",
    }.items():
        if name in config:
            values[name] = config[name]
        elif env_name in os.environ:
            values[name] = os.environ[env_name]
    return GenerationOptions(**values).model_dump()
