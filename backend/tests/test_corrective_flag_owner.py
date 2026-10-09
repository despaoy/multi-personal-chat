"""Corrective feature configuration never silently changes retrieval mode."""
from unittest.mock import Mock

import pytest

from api import generate, knowledge
from knowledge.grounded_answer.service import GroundedAnswerService


@pytest.mark.parametrize("value", ["misspelled", ""])
async def test_bad_generation_flag_fails_before_index_work(monkeypatch, value):
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", value)
    ensure = Mock(side_effect=AssertionError("index must not be consulted"))
    monkeypatch.setattr(knowledge, "_ensure_vector_index", ensure)
    with pytest.raises(ValueError, match="CORRECTIVE_RAG_ENABLED"):
        await generate._retrieve_rag_bundle("核对规则", 3, {"knowledge_base_id": 7})
    ensure.assert_not_called()


@pytest.mark.parametrize("value,expected", [(None, True), ("true", True), ("off", False), ("misspelled", None), ("", None)])
def test_grounded_flag_preserves_default_and_explicit_modes(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("GROUNDED_ANSWER_CORRECTIVE", raising=False)
    else:
        monkeypatch.setenv("GROUNDED_ANSWER_CORRECTIVE", value)
    if expected is None:
        with pytest.raises(ValueError, match="GROUNDED_ANSWER_CORRECTIVE"):
            GroundedAnswerService()
    else:
        assert GroundedAnswerService().corrective_enabled is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_grounded_setting_owns_the_mode(monkeypatch, enabled):
    monkeypatch.setenv("GROUNDED_ANSWER_CORRECTIVE", "unused-invalid-environment")
    assert GroundedAnswerService(corrective_enabled=enabled).corrective_enabled is enabled
