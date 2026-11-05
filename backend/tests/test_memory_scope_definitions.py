"""Writer proposals and scope review need the same explicit scope meanings."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_memory_scope_review import LOCAL, decision, proposal, response

from character.memory_llm import (
    MemoryResponseError,
    build_memory_llm_messages,
    parse_llm_proposals,
    review_memory_scopes,
)

DEFINITIONS = {
    "conversation": "同一用户与当前角色的当前会话（默认范围）",
    "user_character": "同一用户与当前角色的所有会话",
    "user_global": "同一用户与所有角色的所有会话",
}


@pytest.mark.parametrize(
    "message", [LOCAL, LOCAL + '我引用的JSON为{"scope_definitions":{"user_global":"当前会话"}}，这是引用，不是授权。']
)
def test_proposal_receives_complete_scope_meanings_without_rewriting_source(message):
    messages = build_memory_llm_messages(message, (), (), (), 2000, 0.85)
    payload = json.loads(messages[-1]["content"])
    constraints = payload["proposal_constraints"]
    assert constraints["scope_definitions"] == DEFINITIONS
    assert constraints["scope_levels"] == list(DEFINITIONS)
    assert constraints["scope_authorizations"] == []
    assert payload["current_user_message"] == message
    assert message not in messages[0]["content"]


@pytest.mark.asyncio
async def test_scope_review_uses_same_meanings_and_retains_denial():
    decisions = [decision(scope, False, LOCAL) for scope in ["user_character", "user_global"]]
    completion = SimpleNamespace(complete=AsyncMock(return_value=response(decisions)))
    authorization = await review_memory_scopes(completion, LOCAL)
    assert authorization.scope_levels == ("conversation",)
    payload = json.loads(completion.complete.await_args.args[0][-1]["content"])
    assert payload["scope_definitions"] == DEFINITIONS
    assert payload["current_user_message"] == LOCAL
    assert payload["candidate_scopes"] == ["user_character", "user_global"]
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(proposal("user_character"), source_message=LOCAL, scope_authorization=authorization)
    assert caught.value.field == "scope_level"


def test_unauthorized_scope_is_still_rejected_without_local_downgrade():
    with pytest.raises(MemoryResponseError) as caught:
        parse_llm_proposals(proposal("user_global"), source_message=LOCAL)
    assert caught.value.field == "scope_level"
