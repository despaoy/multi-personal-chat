"""Curated routing uses one runtime and never masks malformed indexed evidence."""

from knowledge.index_lifecycle import default_knowledge_index as index_lifecycle

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_bound_domain_routing import bound_scope, setup_scope
from test_multiscale_card_recall import QUERIES, arrange

from knowledge import chat_retrieval
from knowledge.multiscale_rag import runtime as runtime_module


@pytest.mark.asyncio
async def test_unresolved_binding_uses_one_runtime_for_planning_and_retrieval(tmp_path, monkeypatch):
    rig = setup_scope(tmp_path, monkeypatch)
    get_runtime = Mock(return_value=rig.runtime)
    monkeypatch.setattr(runtime_module, 'get_multiscale_rag_service', get_runtime)
    query = '林栖的登记身份是什么？'
    bundle = await chat_retrieval.retrieve_chat_knowledge(query, 1, None, question_binding=await bound_scope(query, names=()), database=index_lifecycle.db)
    assert bundle['domains'] == [rig.config.domain_id]
    get_runtime.assert_called_once_with()
    rig.generic.assert_not_called()


def test_malformed_indexed_evidence_does_not_become_summary_only(tmp_path):
    rig = arrange(tmp_path)
    rig.service.evidence_by_parent["fact"] = SimpleNamespace()
    with pytest.raises(AttributeError, match="content"):
        rig.service.retrieve(QUERIES["fact"], top_k=1)
