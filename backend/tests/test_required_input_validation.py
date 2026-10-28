"""Request validation is a required API dependency rather than a capability flag."""

from knowledge.index_lifecycle import default_knowledge_index as index_lifecycle

import base64
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


@pytest.mark.parametrize(
    "module,dependency",
    [
        ("api.generate", "db.schemas"),
        ("api.config", "infra.input_validator"),
        ("api.knowledge", "infra.input_validator"),
        ("api.training", "infra.input_validator"),
        ("app.main", "infra.input_validator"),
    ],
)
def test_missing_validation_dependency_prevents_component_import(module, dependency, tmp_path):
    source = """import builtins,importlib,json,logging,sys
logging.disable(logging.CRITICAL)
original=builtins.__import__
failure=ImportError('synthetic required input dependency unavailable')
def blocked(name,*args,**kwargs):
 if name==sys.argv[2]:raise failure
 return original(name,*args,**kwargs)
builtins.__import__=blocked
try:
 importlib.import_module(sys.argv[1])
except ImportError as error:
 print(json.dumps(dict(original_dependency_error=error is failure)))
else:
 print(json.dumps(dict(original_dependency_error=False)))
"""
    env = dict(
        os.environ,
        ENVIRONMENT="test",
        USE_POSTGRESQL="false",
        DATABASE_PATH=str(tmp_path / "isolated.sqlite"),
        DATABASE_URL="postgresql+asyncpg://unused@127.0.0.1:1/unused",
        JWT_SECRET="required-input-synthetic-key-more-than-32-characters",
        ENCRYPTION_KEY=base64.urlsafe_b64encode(b"v" * 32).decode(),
    )
    process = subprocess.run(
        [sys.executable, "-c", source, module, dependency], capture_output=True, text=True, env=env, timeout=30
    )
    assert process.returncode == 0
    assert json.loads(process.stdout)["original_dependency_error"] is True


async def test_accepted_message_cannot_change_before_context_model_or_archive(monkeypatch):
    from test_turn_completion_admission import generation_environment

    _gen, runtime, request, prepare, generation, storage = generation_environment(monkeypatch, "openai_compat")
    original = request.message
    with pytest.raises(ValidationError) as raised:
        request.message = ""
    assert raised.value.errors()[0]["type"] == "frozen_field"
    assert request.message == original
    prepare.assert_not_awaited()
    generation.assert_not_awaited()
    storage.assert_not_awaited()
    assert runtime.active == runtime.reserved == 0
    await runtime.shutdown(timeout=1)


async def test_config_limits_are_checked_before_any_write(monkeypatch):
    from api import config

    run_db = AsyncMock(side_effect=AssertionError("Invalid configuration must not reach storage"))
    monkeypatch.setattr(config, "run_db", run_db)
    request = SimpleNamespace(json=AsyncMock(return_value={"k" * 101: "synthetic value"}))
    with pytest.raises(HTTPException) as raised:
        await config.update_config(request, {"role": "admin"})
    assert raised.value.status_code == 422
    run_db.assert_not_awaited()


async def test_document_limits_are_checked_before_storage_or_indexing(monkeypatch):
    from api import knowledge
    from db.schemas import KnowledgeDocumentCreate
    run_db = AsyncMock(side_effect=AssertionError('Invalid document must not reach storage or indexing'))
    monkeypatch.setattr(knowledge, 'run_db', run_db)
    with pytest.raises(HTTPException) as raised:
        await knowledge.create_knowledge_document(KnowledgeDocumentCreate(title='', content='完整的合成正文。'), {'role': 'admin'}, database=index_lifecycle.db)
    assert raised.value.status_code == 422
    run_db.assert_not_awaited()
