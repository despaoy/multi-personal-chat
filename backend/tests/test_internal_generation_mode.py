"""Internal generation mode must not persist chat rows or model invocations."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from db.schemas import MessageRequest


def _request():
    return MessageRequest(message="tool-json", sessionType="private", sessionId="claw")


def _install_common(monkeypatch, gen):
    monkeypatch.setattr(gen, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(gen, "response_cache", None)
    monkeypatch.setattr(gen, "db", SimpleNamespace(config={}, loras=[]))
    monkeypatch.setattr(gen, "circuit_breaker_registry", None)
    saved = []
    recorded = []
    monkeypatch.setattr(gen, "_save_message", _record_save(saved))
    monkeypatch.setattr(gen, "_record_model_invocation", _record_invocation(recorded))
    return saved, recorded


def _record_save(saved):
    async def fake(*args, **kwargs):
        saved.append(args)
    return fake


def _record_invocation(recorded):
    async def fake(*args, **kwargs):
        recorded.append(args)
    return fake


@pytest.mark.asyncio
async def test_internal_mode_success_does_not_persist(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm

    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())

    async def fake_vllm(*args, **kwargs):
        return "tool-reply", False, {}

    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    result = await gen._generate_reply_impl(
        _request(),
        persist_message=False,
        enable_rag=False,
        record_invocation=False,
    )

    assert result.reply == "tool-reply"
    assert saved == []
    assert recorded == []


@pytest.mark.asyncio
async def test_internal_mode_vllm_failure_does_not_persist(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm

    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

        def set_lora_adapter(self, value):
            pass

        async def async_generate(self, **kwargs):
            return "fallback-reply", 0.01

        def get_status(self):
            return {"currentProvider": "vllm", "providers": {"vllm": {"modelName": "fake"}}}

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(2))

    async def fake_vllm(*args, **kwargs):
        raise RuntimeError("vllm down")

    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    result = await gen._generate_reply_impl(
        _request(),
        persist_message=False,
        enable_rag=False,
        record_invocation=False,
    )

    assert result.reply == "fallback-reply"
    assert saved == []
    assert recorded == []


@pytest.mark.asyncio
async def test_internal_mode_model_manager_failure_does_not_record_invocation(monkeypatch):
    import api.generate as gen
    from inference import model_manager as mm

    saved, recorded = _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

        def set_lora_adapter(self, value):
            pass

        async def async_generate(self, **kwargs):
            raise RuntimeError("model manager down")

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(2))

    async def fake_vllm(*args, **kwargs):
        raise RuntimeError("vllm down")

    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    with pytest.raises(HTTPException):
        await gen._generate_reply_impl(
            _request(),
            persist_message=False,
            enable_rag=False,
            record_invocation=False,
        )

    assert saved == []
    assert recorded == []


def _async_return(value):
    async def fake(*args, **kwargs):
        return value
    return fake


@pytest.mark.asyncio
async def test_message_db_reaches_save_and_invocation_record(monkeypatch):
    """容器注入的数据库必须贯穿消息保存与模型调用记录。

    只有人物服务走容器库、消息仍写全局库时，自定义容器实例的
    上一轮消息写入全局库，下一轮人物历史从容器库读不到，历史断裂。
    """
    import api.generate as gen
    from inference import model_manager as mm

    _install_common(monkeypatch, gen)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())

    async def fake_vllm(*args, **kwargs):
        return "reply", False, {}

    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    captured = {}

    async def fake_save(request, reply, model_name, lora_name, cost_time, *, database=None):
        captured["save_db"] = database
        return True

    async def fake_record(request, model_name, lora_name, cost_time, **kwargs):
        captured["record_db"] = kwargs.get("database")

    monkeypatch.setattr(gen, "_save_message", fake_save)
    monkeypatch.setattr(gen, "_record_model_invocation", fake_record)

    sentinel_db = SimpleNamespace(config={}, loras=[])
    result = await gen._generate_reply_impl(
        _request(),
        persist_message=True,
        enable_rag=False,
        record_invocation=True,
        message_db=sentinel_db,
    )

    assert result.reply == "reply"
    assert captured["save_db"] is sentinel_db
    assert captured["record_db"] is sentinel_db


@pytest.mark.asyncio
async def test_save_message_and_invocation_use_injected_database(monkeypatch):
    """_save_message/_record_model_invocation 必须写入注入的数据库对象。"""
    import api.generate as gen

    calls = []

    class FakeDB:
        def add_message(self, record):
            calls.append(("message", record))
            return 1

        def add_model_invocation(self, record):
            calls.append(("invocation", record))
            return 1

    sentinel_db = FakeDB()
    request = _request()

    ok = await gen._save_message(
        request, "reply", "test-model", "default", 0.1, database=sentinel_db
    )
    assert ok is True
    assert calls[-1][0] == "message"
    assert calls[-1][1]["message"] == request.message

    await gen._record_model_invocation(
        request, "test-model", "default", 0.1, database=sentinel_db
    )
    assert calls[-1][0] == "invocation"
    # 两次写入都落在注入的数据库上（而非全局单例）
    assert len(calls) == 2


def test_lora_path_resolution_uses_injected_loras(monkeypatch):
    """本地 LoRA 回退路径查找必须使用容器数据库读出的 LoRA 列表。

    默认查全局 db.loras 时，自定义容器下 LoRA 列表来自容器库、
    路径却在全局库查不到，会静默退回基座模型。
    """
    from db import database as db_mod

    monkeypatch.setattr(
        db_mod, "refresh_lora_dir_map", lambda: {"kisaki": "/loras/kisaki"}
    )
    # 全局库没有 id=42 的 LoRA；容器库有
    monkeypatch.setattr(db_mod, "db", SimpleNamespace(loras=[]))
    container_loras = [{"id": 42, "name": "kisaki"}]

    # 注入容器列表：能解析到路径
    assert db_mod.get_lora_path_by_id(42, loras=container_loras) == "/loras/kisaki"
    # 不注入时回退全局库：查不到，返回 None
    assert db_mod.get_lora_path_by_id(42) is None


def test_resolve_kb_id_uses_injected_database(monkeypatch):
    """RAG 知识库映射的数据库回退必须使用注入的数据库。

    自定义容器下查全局库会拿到错误的知识库 ID，RAG 检索过滤到
    其他实例的知识库。
    """
    import api.generate as gen

    class GlobalDB:
        def get_knowledge_bases(self):
            return [{"id": 99, "name": "全局知识库"}]

    class ContainerDB:
        def get_knowledge_bases(self):
            return [{"id": 7, "name": "游戏攻略"}]

    monkeypatch.setattr(gen, "db", GlobalDB())
    # 容器注入的数据库优先：返回容器库的 ID
    assert gen._resolve_kb_id("游戏攻略", database=ContainerDB()) == 7
    # 同名知识库在全局库中不存在：不注入时查不到
    assert gen._resolve_kb_id("游戏攻略") is None


def test_resolve_kb_id_container_db_bypasses_shared_config(monkeypatch):
    """注入容器数据库时不得读取进程共享的训练配置映射。

    intent_classifier_model/config.json 是全局训练产物：若其中保存了
    同名知识库的旧 ID（99），先读它会让容器实例绕过自己的数据库，
    RAG 过滤到错误的知识库。容器注入路径必须只查容器数据库。
    """
    import api.generate as gen

    class GlobalDB:
        def get_knowledge_bases(self):
            return [{"id": 88, "name": "全局知识库"}]

    class ContainerDB:
        def get_knowledge_bases(self):
            return [{"id": 7, "name": "游戏攻略"}]

    # 共享配置含同名知识库的旧 ID
    monkeypatch.setattr(
        gen, "_read_shared_kb_config_mapping", lambda: {"游戏攻略": 99}
    )
    monkeypatch.setattr(gen, "db", GlobalDB())

    # 容器注入：即使共享配置有旧 ID，也必须返回容器库的真实 ID
    assert gen._resolve_kb_id("游戏攻略", database=ContainerDB()) == 7
    # 容器库查不到的名字：不得回退共享配置或全局库
    assert gen._resolve_kb_id("不存在的库", database=ContainerDB()) is None

    # 不注入时（全局默认路径）共享配置优先的既有行为保持不变
    assert gen._resolve_kb_id("游戏攻略") == 99


@pytest.mark.asyncio
async def test_model_manager_fallback_uses_character_context(monkeypatch):
    """vLLM 不可用回退到模型管理器时，必须使用编译后的角色上下文。

    回退路径不得退化为"原始消息 + request.history"：关系/情景/决策
    要进系统提示词，长期记忆进用户消息的不可信参考区，数据库历史
    在 request.history 为空时兜底，否则生成成功后回写的人物状态
    与实际喂给模型的上下文脱节。
    """
    import api.generate as gen
    from inference import model_manager as mm

    saved, recorded = _install_common(monkeypatch, gen)
    # 激活映射了人物画像的 kisaki LoRA
    monkeypatch.setattr(
        gen,
        "db",
        SimpleNamespace(config={}, loras=[{"name": "kisaki", "status": "active", "id": 1}]),
    )

    fake_prepared = SimpleNamespace(
        character_id="tsukiyashiro_kisaki",
        compiled=SimpleNamespace(
            profile_context="人物画像",
            dynamic_context="关系动态：熟悉阶段",
            reference_context="长期记忆：用户说自己叫小明",
        ),
        history=({"role": "user", "content": "数据库历史消息"},),
    )

    async def fake_prepare(request, character_id, *, character_service=None):
        assert character_id == "tsukiyashiro_kisaki"
        return fake_prepared

    monkeypatch.setattr(gen, "_prepare_character_turn", fake_prepare)

    captured = {}

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

        def set_lora_adapter(self, value):
            pass

        async def async_generate(self, **kwargs):
            captured.update(kwargs)
            return "fallback-reply", 0.01

        def get_status(self):
            return {"currentProvider": "vllm", "providers": {"vllm": {"modelName": "fake"}}}

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())
    # vLLM 整体不可用：跳过 vLLM 分支直接进入模型管理器回退
    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(False))
    monkeypatch.setattr(gen, "_vllm_client", None)
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(2))

    request = MessageRequest(
        message="我们聊点什么",
        sessionType="private",
        sessionId="s1",
        senderName="小明",
        history=[],
    )
    result = await gen._generate_reply_impl(
        request,
        persist_message=False,
        enable_rag=False,
        record_invocation=False,
    )

    assert result.reply == "fallback-reply"
    assert saved == []
    assert recorded == []

    session_history = captured["session_history"]
    # 系统提示词包含编译后的关系动态，且不含长期记忆（不可信数据不进系统区）
    system_message = session_history[0]
    assert system_message["role"] == "system"
    assert "关系动态：熟悉阶段" in system_message["content"]
    assert "长期记忆" not in system_message["content"]
    # request.history 为空时使用数据库加载的历史
    assert any(m["content"] == "数据库历史消息" for m in session_history)
    # 长期记忆进入最终用户消息的不可信参考区
    prompt = captured["prompt"]
    assert "长期记忆：用户说自己叫小明" in prompt
    assert "我们聊点什么" in prompt


@pytest.mark.asyncio
async def test_injected_character_service_reaches_prepare_and_complete(monkeypatch):
    """HTTP 层注入的容器角色服务必须贯穿准备与回写两个阶段。

    create_app(custom_container) 的生成链路若回退全局默认服务，
    人物记忆/关系会读写全局数据库，与管理接口（走容器）脱节；
    且准备与回写必须使用同一服务实例（同一容器数据库）。
    """
    import api.generate as gen
    from inference import model_manager as mm

    _install_common(monkeypatch, gen)
    monkeypatch.setattr(
        gen,
        "db",
        SimpleNamespace(config={}, loras=[{"name": "kisaki", "status": "active", "id": 1}]),
    )

    # 消息保存成功，让回写路径执行
    async def fake_save(*args, **kwargs):
        return True

    monkeypatch.setattr(gen, "_save_message", fake_save)

    container_service = object()  # 哨兵：HTTP 层注入的服务实例
    seen = {}

    fake_prepared = SimpleNamespace(
        character_id="tsukiyashiro_kisaki",
        compiled=SimpleNamespace(
            profile_context="人物画像",
            dynamic_context="关系动态",
            reference_context="长期记忆",
        ),
        history=(),
    )

    async def fake_prepare(request, character_id, *, character_service=None):
        seen["prepare_service"] = character_service
        return fake_prepared

    async def fake_complete(prepared, request, reply, *, character_service=None):
        seen["complete_service"] = character_service

    monkeypatch.setattr(gen, "_prepare_character_turn", fake_prepare)
    monkeypatch.setattr(gen, "_complete_character_turn", fake_complete)

    class FakeManager:
        _current_provider = SimpleNamespace(value="vllm")

    monkeypatch.setattr(mm, "get_model_manager", lambda: FakeManager())

    async def fake_vllm(*args, **kwargs):
        return "reply", False, {}

    monkeypatch.setattr(gen, "_ensure_vllm", _async_return(True))
    monkeypatch.setattr(gen, "_vllm_client", object())
    monkeypatch.setattr(gen, "_generate_with_vllm", fake_vllm)

    request = MessageRequest(message="你好", sessionType="private", sessionId="s1")
    result = await gen._generate_reply_impl(
        request,
        persist_message=True,
        enable_rag=False,
        record_invocation=False,
        character_service=container_service,
    )

    assert result.reply == "reply"
    # 准备与回写必须使用同一注入服务（读写同一容器数据库）
    assert seen["prepare_service"] is container_service
    assert seen["complete_service"] is container_service
