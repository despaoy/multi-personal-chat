"""Invalid feature flags must never silently disable context processing."""
from unittest.mock import Mock

import pytest
from services.character_context import CharacterContextService

from character.contextual_policy import create_contextual_policy
from character.evidence_selector import create_evidence_selector
from character.memory_llm import MemoryLlmConfig
from character.semantic_review_adapter import SemanticReviewSettings
from infra.environment import read_bool

FLAGS=['MEMORY_LLM_ENABLED','CONTEXTUAL_MEMORY_SELECTION_ENABLED','CONTEXTUAL_DECISION_POLICY_ENABLED','DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED','MEMORY_SOURCE_RECALL_ENABLED']


def enabled(name):
    if name=='MEMORY_LLM_ENABLED':
        return MemoryLlmConfig.from_env().enabled
    if name=='CONTEXTUAL_MEMORY_SELECTION_ENABLED':
        return create_evidence_selector() is not None
    if name=='CONTEXTUAL_DECISION_POLICY_ENABLED':
        return create_contextual_policy() is not None
    if name=='DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED':
        return SemanticReviewSettings.from_env().enabled
    return CharacterContextService(Mock(),Mock(),Mock(),memory_service=Mock())._source_recall_enabled


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in FLAGS:
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('MEMORY_LLM_BASE_URL','http://fixture.invalid')
    monkeypatch.setenv('MEMORY_LLM_MODEL','fixture')


@pytest.mark.parametrize('name',FLAGS)
@pytest.mark.parametrize('value',['','tru','enabled','2','private-invalid-value'])
def test_invalid_flag_raises_without_echo(monkeypatch,name,value):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError,match=name) as caught:
        enabled(name)
    assert 'private-invalid-value' not in str(caught.value)


@pytest.mark.parametrize('name',FLAGS)
@pytest.mark.parametrize('value,expected',[(None,False),(' false ',False),(' TRUE ',True)])
def test_defaults_and_explicit_flags(monkeypatch,name,value,expected):
    if value is not None:
        monkeypatch.setenv(name,value)
    assert enabled(name) is expected


@pytest.mark.parametrize('value,expected',[('1',True),('yes',True),('on',True),('0',False),('no',False),('off',False)])
def test_existing_boolean_spellings(value,expected):
    assert read_bool({'FLAG':value},'FLAG') is expected


@pytest.mark.parametrize('override',[False,True])
def test_explicit_source_setting_wins_over_environment(monkeypatch,override):
    monkeypatch.setenv('MEMORY_SOURCE_RECALL_ENABLED','invalid-unused-setting')
    service=CharacterContextService(Mock(),Mock(),Mock(),memory_service=Mock(),source_recall_enabled=override)
    assert service._source_recall_enabled is override
