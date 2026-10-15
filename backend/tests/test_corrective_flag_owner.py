"""The independent grounded-answer corrective option is explicit."""

import pytest

from knowledge.grounded_answer.service import GroundedAnswerService


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
