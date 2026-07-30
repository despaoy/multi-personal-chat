"""Regression coverage for the architecture hardening pass."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest


def test_response_cache_identity_isolated_by_lora_and_config(monkeypatch):
    from api import generate
    from db.schemas import MessageRequest

    config = {"temperature": 0.7, "maxTokens": 128, "useKnowledgeBase": False}
    monkeypatch.setattr(generate, "db", SimpleNamespace(config=config))
    request = MessageRequest(message="hello", sessionId="session-1", platform="qq")

    base_keys = generate._response_cache_keys(request, "default")
    lora_keys = generate._response_cache_keys(request, "kisaki")
    assert base_keys[:2] != lora_keys[:2]

    config["temperature"] = 0.2
    changed_keys = generate._response_cache_keys(request, "default")
    assert base_keys[:2] != changed_keys[:2]


@pytest.mark.asyncio
async def test_raise_degradation_never_fabricates_a_model_success():
    from infra.circuit_breaker import (
        CircuitBreaker,
        CircuitOpenError,
        DegradationMode,
    )

    breaker = CircuitBreaker(
        name="model",
        failure_threshold=1,
        recovery_timeout=60,
        degradation_mode=DegradationMode.RAISE,
    )

    async def fail():
        raise RuntimeError("model down")

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    with pytest.raises(CircuitOpenError):
        await breaker.call(fail)


def test_rag_confidence_uses_absolute_score_not_rank_normalization():
    from knowledge.rag_helper import RAGHelper

    results = [
        {"score": 0.12, "normalized_score": 1.0},
        {"score": 0.08, "normalized_score": 0.5},
    ]
    confidence = RAGHelper.compute_confidence(object.__new__(RAGHelper), results)
    assert 0.0 < confidence < 0.3


def test_rag_citations_keep_knowledge_base_filters():
    from knowledge.rag_helper import RAGHelper

    helper = object.__new__(RAGHelper)
    captured = {}

    def retrieve(query, top_k=None, filters=None):
        captured["filters"] = filters
        return [{"title": "doc", "content": "evidence", "score": 0.9}]

    helper.retrieve_context = retrieve
    result = helper.retrieve_with_citations(
        "question", top_k=1, filters={"knowledge_base_id": 7}
    )
    assert captured["filters"] == {"knowledge_base_id": 7}
    assert result["abstained"] is False


def test_experiment_dataclasses_are_json_ready():
    from api.experiments import _serialize_results

    @dataclass
    class Result:
        variant: str
        score: float

    assert _serialize_results([Result("hybrid", 0.9)]) == [
        {"variant": "hybrid", "score": 0.9}
    ]


def test_training_truncation_preserves_supervised_response_tokens():
    from training.trainer import LoRATrainer

    trainer = object.__new__(LoRATrainer)
    trainer.config = SimpleNamespace(max_seq_length=5, truncation_direction="left")
    full_ids, prompt_len = trainer._truncate_preserving_response(
        [1, 2, 3, 4, 5, 90, 91, 92], [1, 2, 3, 4, 5]
    )
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    assert full_ids == [4, 5, 90, 91, 92]
    assert labels[-3:] == [90, 91, 92]


# ============================================================
# C-S1 fix: RBAC admin 依赖回归测试
# 验证 get_current_admin 在 admin/user/DB不可达 三种场景下的行为
# ============================================================


class _FakeRequest:
    """最小化的 Request 替身，仅满足 get_current_user/get_current_admin 的访问需求。"""

    def __init__(self, payload: dict | None):
        self.state = SimpleNamespace(jwt_payload=payload)
        self.headers = {}
        self.cookies = {}


@pytest.mark.asyncio
async def test_get_current_admin_allows_admin_role(monkeypatch):
    """admin 用户应通过 get_current_admin 校验"""
    from app import dependencies

    payload = {"sub": "admin_user", "user_id": 1, "role": "admin", "jti": "jti-admin"}
    request = _FakeRequest(payload)

    # 绕过 token 黑名单检查
    monkeypatch.setattr(
        "api.auth.is_token_revoked", lambda jti: False, raising=False
    )
    # DB 复核仍返回 admin
    monkeypatch.setattr(
        "db.adapter.db.get_user_by_username",
        lambda username: {"id": 1, "username": username, "role": "admin"},
        raising=False,
    )

    user = await dependencies.get_current_admin(request)
    assert user["role"] == "admin"
    assert user["username"] == "admin_user"


@pytest.mark.asyncio
async def test_get_current_admin_rejects_non_admin_role(monkeypatch):
    """普通 user 应被 get_current_admin 拒绝（403），即使 token 中 role=admin"""
    from fastapi import HTTPException
    from app import dependencies

    # token 中持 admin，但 DB 已降级为 user（模拟管理员被降级后旧 token 仍持 admin）
    payload = {"sub": "demoted_user", "user_id": 2, "role": "admin", "jti": "jti-demoted"}
    request = _FakeRequest(payload)

    monkeypatch.setattr(
        "api.auth.is_token_revoked", lambda jti: False, raising=False
    )
    monkeypatch.setattr(
        "db.adapter.db.get_user_by_username",
        lambda username: {"id": 2, "username": username, "role": "user"},
        raising=False,
    )

    with pytest.raises(HTTPException) as exc:
        await dependencies.get_current_admin(request)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_get_current_admin_rejects_when_db_unreachable(monkeypatch):
    """DB 不可达时按最小权限原则拒绝（503），避免误授权"""
    from fastapi import HTTPException
    from app import dependencies

    payload = {"sub": "admin_user", "user_id": 1, "role": "admin", "jti": "jti-db-down"}
    request = _FakeRequest(payload)

    monkeypatch.setattr(
        "api.auth.is_token_revoked", lambda jti: False, raising=False
    )

    def _raise(_username):
        raise RuntimeError("DB connection lost")

    monkeypatch.setattr(
        "db.adapter.db.get_user_by_username", _raise, raising=False
    )

    with pytest.raises(HTTPException) as exc:
        await dependencies.get_current_admin(request)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_get_current_user_returns_role_from_payload(monkeypatch):
    """get_current_user 应返回 JWT payload 中的 role，缺失时默认 user"""
    from app import dependencies

    monkeypatch.setattr(
        "api.auth.is_token_revoked", lambda jti: False, raising=False
    )

    # 1. payload 含 role
    payload = {"sub": "u1", "user_id": 1, "role": "admin", "jti": "jti-1"}
    user = await dependencies.get_current_user(_FakeRequest(payload))
    assert user["role"] == "admin"

    # 2. payload 缺失 role（旧 token）→ 默认 "user"
    payload_old = {"sub": "u2", "user_id": 2, "jti": "jti-2"}
    user_old = await dependencies.get_current_user(_FakeRequest(payload_old))
    assert user_old["role"] == "user"


def test_create_access_token_includes_role_in_payload():
    """JWT 应包含 role 字段，便于 get_current_user 无需 DB 查询即可返回角色"""
    from app.config import create_access_token, verify_token, JWT_SECRET, JWT_ALGORITHM
    import jwt as pyjwt

    token = create_access_token("alice", 42, role="admin")
    payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    assert payload["role"] == "admin"
    assert payload["sub"] == "alice"
    assert payload["user_id"] == 42

    # 默认 role 为 user
    token_default = create_access_token("bob", 43)
    payload_default = pyjwt.decode(token_default, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    assert payload_default["role"] == "user"
