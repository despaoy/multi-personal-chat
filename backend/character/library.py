"""Portable user-authored character settings; no model or account credentials."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExampleMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class CharacterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    persona: str = Field(min_length=1, max_length=16000)
    greeting: str = Field(min_length=1, max_length=8000)
    scenario: str = Field(default="", max_length=8000)
    relationships: str = Field(default="", max_length=8000)
    examples: list[ExampleMessage] = Field(default_factory=list, max_length=40)
    reference_text: str = Field(default="", max_length=40000)


class CharacterPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["multipersonal-character-v1"] = "multipersonal-character-v1"
    settings: CharacterSettings
