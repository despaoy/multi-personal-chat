import sys
from types import SimpleNamespace

import pytest

from training.preference_compat import preference_length_kwargs, resolve_preference_backend


def test_missing_orpo_fails_at_capability_check(monkeypatch):
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(DPOTrainer="trainer", DPOConfig="config"))
    assert resolve_preference_backend("dpo") == ("trainer", "config")
    with pytest.raises(RuntimeError, match="does not expose ORPO"):
        resolve_preference_backend("orpo")
    with pytest.raises(ValueError):
        resolve_preference_backend("invented")


def test_old_trl_prompt_limit_matches_validated_budget():
    def config(max_length=512, max_prompt_length=256):
        pass

    assert preference_length_kwargs(config, max_length=4096, max_prompt_length=3072) == {
        "max_length": 4096,
        "max_prompt_length": 3072,
    }


def test_new_trl_only_total_length_is_passed():
    def config(max_length=1024):
        pass

    assert preference_length_kwargs(config, max_length=4096, max_prompt_length=3072) == {"max_length": 4096}
    with pytest.raises(ValueError):
        preference_length_kwargs(config, max_length=512, max_prompt_length=512)
