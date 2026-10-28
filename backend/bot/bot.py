"""
QQ自动回复机器人 - 多LoRA角色风格
基于 NoneBot2 接入统一后端生成服务
支持私聊和@消息识别

集成架构：
  Bot → db.adapter（统一数据库访问）→ SQLite/PostgreSQL
  Bot → ChatGenerationService + InferenceRuntime → 已配置的模型与检索服务
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from nonebot import on_command, on_message
from nonebot.adapters.onebot.v11 import (
    Bot,
    MessageEvent,
    Message,
    GroupMessageEvent,
    PrivateMessageEvent,
)
from nonebot.log import logger
from nonebot.exception import FinishedException
from bot.tools import execute_tool, ToolCall, ToolError, TOOLS
from pydantic import ValidationError

# 加载环境变量 - 优先加载backend/.env，回退到bot/.env
_backend_env = Path(__file__).parent.parent / ".env"
_bot_env = Path(__file__).parent / ".env"
load_dotenv(_backend_env)
if _bot_env.exists():
    load_dotenv(_bot_env, override=False)

# 确保backend目录在sys.path中（db.adapter等模块需要）
_BACKEND_DIR = str(Path(__file__).parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ============================================
# 统一数据库访问 - 通过db.adapter
# ============================================
from db.adapter import db
from character.models import UserScope, build_user_scope

# 统一生成链路：NoneBot 不再自行组装 prompt/RAG/失败回退
from api.generate import generate_reply_core
from db.schemas import MessageRequest
from infra.environment import read_positive_float
from infra.concurrency_control import (
    InferenceQueueFull,
    RateLimitExceeded,
    inference_runtime,
)

# ============================================
# 消息去重（幂等设计）
# ============================================
_DEDUP_TTL = 3600  # 1小时TTL


class MessageDeduplicationError(RuntimeError):
    """Message processing must stop when shared deduplication is unverified."""


class SessionReplyCheckError(RuntimeError):
    """The session reply setting could not be read."""


async def _is_duplicate_message(message_id: str) -> bool:
    """Reserve one shared message identity; service failures are not misses."""
    try:
        from cache.redis_client import get_async_redis

        redis = await get_async_redis()
        was_set = await redis.set(f"dedup:msg:{message_id}", "1", nx=True, ex=_DEDUP_TTL)
    except Exception as exc:
        raise MessageDeduplicationError("Shared message deduplication failed") from exc
    if was_set is True:
        return False
    if was_set is None or was_set is False:
        return True
    raise MessageDeduplicationError("Shared message deduplication returned an invalid status")

# ============================================
# 配置 - 通过db.adapter读取
# ============================================

# Same-process readers share one bounded snapshot. A separate legacy bot
# process observes external changes when its own TTL expires.
def _get_db_config() -> Dict[str, Any]:
    from cache.config_cache import read_config

    return read_config(db)[0]


class Config:
    """机器人配置 - 环境变量项为静态属性，数据库项通过@property动态读取（60秒 TTL 缓存）"""
    # 环境变量配置（静态，启动时读取一次）
    SUPERUSERS = [str(u) for u in json.loads(os.getenv("SUPERUSERS", "[]"))]

    # 以下配置通过 @property 动态读取 db.config，独立进程在约一分钟内重新读取配置
    @property
    def BOT_NAME(self):
        return _get_db_config().get('botName', os.getenv("NICKNAME", "胡桃"))

    @property
    def REPLY_DELAY(self):
        return float(_get_db_config().get('replyDelay', os.getenv("REPLY_DELAY", "0.8")))

config = Config()

# Per-process resets retain only the archive position, never copied speech.
_history_resets: dict[tuple[str, ...], int] = {}
claw_sessions: Dict[str, bool] = {}

# Empty selection follows backend configuration; "default" explicitly selects its base model.
_selected_lora = ""


class LoraSelectionError(RuntimeError):
    """A requested bot-wide adapter cannot be selected."""


async def _select_lora(target: str) -> str:
    """Select a registered adapter; the shared backend owns all model weights."""
    global _selected_lora
    if target == "default":
        _selected_lora = target
        return target

    from inference.lora_utils import resolve_lora_served_name
    from inference.model_manager import get_model_manager

    loras = await asyncio.to_thread(db.get_loras)
    matches = [item for item in loras if item["name"] == target]
    if not matches:
        served_name = resolve_lora_served_name(target)
        matches = [item for item in loras if resolve_lora_served_name(item["name"]) == served_name]
    if len(matches) != 1:
        raise LoraSelectionError("LoRA 不存在或别名对应多项，请使用 /lora 查看登记名称。")
    selected = matches[0]
    provider = get_model_manager().get_current_provider().name
    if provider == "vllm":
        from inference.vllm_client import get_enabled_vllm_client

        client = await get_enabled_vllm_client()
        if client is None:
            raise LoraSelectionError("vLLM 服务未启用，请先在管理页配置。")
        if resolve_lora_served_name(selected["name"]) not in await client.list_loras():
            raise LoraSelectionError("LoRA 尚未加载，请先在管理页加载后再选择。")
    elif provider == "transformers_peft":
        from db.database import get_lora_path_by_id

        path = await asyncio.to_thread(get_lora_path_by_id, selected["id"], loras=loras)
        if path is None:
            raise LoraSelectionError("LoRA 文件不存在，请先在管理页配置适配器。")
    else:
        raise LoraSelectionError("当前提供商不支持 LoRA，可用 /lora default 选择已配置的基础模型。")
    _selected_lora = selected["name"]
    return _selected_lora


def _get_char_name(lora_name: str) -> str:
    from inference.lora_registry import get_char_name
    from inference.lora_utils import resolve_lora_served_name

    if lora_name in {"", "default"}:
        return config.BOT_NAME
    return get_char_name(resolve_lora_served_name(lora_name))


def is_superuser(event: MessageEvent) -> bool:
    user_id = str(event.user_id)
    return user_id in config.SUPERUSERS


async def _unified_generate(prompt: str, session_id: str, lora_name: str) -> str:
    """Run any NoneBot-side generation through the shared ChatGenerationService."""
    msg = MessageRequest(
        message=prompt,
        sessionType="private",
        conversationType="private",
        sessionId=session_id,
        sessionName="nonebot",
        userId="nonebot",
        userName="nonebot",
        senderName="nonebot",
        platform="qq",
        adapter="nonebot",
        conversationId=session_id,
        senderId="nonebot",
        loraName=lora_name,
        traceId="",
    )
    result = await inference_runtime.submit(
        lambda: generate_reply_core(
            msg,
            current_user={"username": "nonebot", "user_id": 0},
            persist_message=False,
            enable_rag=False,
            invocation_purpose="tool_generation",
            message_db=db,
        ),
        session_id=session_id,
        priority=inference_runtime.priority_for("nonebot", "private"),
        timeout=read_positive_float(os.environ, "MODEL_INFERENCE_TIMEOUT", 180.0),
    )
    return result.reply


# ============================================
# 消息处理
# ============================================
async def should_reply(event: MessageEvent) -> bool:
    """判断是否应该回复该消息"""
    # 检查该会话是否启用了机器人（通过db.adapter）
    is_group = isinstance(event, GroupMessageEvent)
    session_id = str(event.group_id) if is_group else str(event.user_id)
    conversation_type = "group" if is_group else "private"
    try:
        enabled = await asyncio.to_thread(
            db.is_session_bot_enabled,
            session_id,
            "qq",
            session_id,
            conversation_type,
        )
    except Exception as exc:
        raise SessionReplyCheckError("Session reply setting read failed") from exc
    if not enabled:
        logger.info(f"会话 {session_id} 机器人已关闭，跳过")
        return False

    # 私聊消息总是回复
    if isinstance(event, PrivateMessageEvent):
        logger.info("私聊消息，直接回复")
        return True

    # 群聊消息
    if isinstance(event, GroupMessageEvent):
        message_text = str(event.message)
        logger.info(f"群聊消息: 字符数={len(message_text)}")

        if event.is_tome():
            logger.info("检测到@机器人，回复")
            return True
        if config.BOT_NAME in message_text:
            logger.info("检测到包含机器人名称，回复")
            return True
        trigger_words = ["你好", "在吗", "有人吗", config.BOT_NAME, "@ad"]
        char_name = _get_char_name(_selected_lora)
        if char_name != config.BOT_NAME:
            trigger_words.append(char_name)
        for word in trigger_words:
            if word in message_text:
                logger.info(f"检测到触发词'{word}'，回复")
                return True

        logger.info("不满足回复条件")

    return False


def _message_scope(event: MessageEvent) -> UserScope:
    private = isinstance(event, PrivateMessageEvent)
    return build_user_scope(
        "qq", "nonebot", str(event.user_id),
        str(event.user_id) if private else str(event.group_id),
        "private" if private else "group",
    )


async def process_message(event: MessageEvent) -> str:
    """处理消息并生成回复。

    NoneBot 只负责消息接收/回复条件/平台适配；模型生成统一走
    ChatGenerationService + InferenceRuntime，与 API/AstrBot 同一条链路。
    """
    user_message = str(event.message)

    if not user_message.strip():
        return "嗯？怎么不说话呀？"

    logger.info(f"收到消息: 字符数={len(user_message)}")

    scope = _message_scope(event)
    conversation_id = scope.conversation_id
    session_id = conversation_id
    conversation_type = scope.conversation_type
    sender_id = scope.sender_id
    sender_obj = getattr(event, "sender", None)
    sender_name = str(
        getattr(sender_obj, "card", None)
        or getattr(sender_obj, "nickname", None)
        or event.user_id
    )

    start_time = time.time()
    lora_name = _selected_lora
    logger.info("开始处理消息: LoRA={}", lora_name or "backend-config")

    await inference_runtime.check_rate_limits("qq:nonebot", f"{conversation_type}:{conversation_id}", sender_id)
    msg = MessageRequest(
        message=user_message,
        sessionType=conversation_type,
        conversationType=conversation_type,
        sessionId=session_id,
        sessionName=sender_name,
        userId=sender_id,
        userName=sender_name,
        senderName=sender_name,
        platform="qq",
        adapter="nonebot",
        conversationId=conversation_id,
        senderId=sender_id,
        loraName=lora_name,
        traceId="",
    )

    async def _queued_generation():
        # Resolve the reset after earlier queued turns and clear commands finish.
        msg._history_after_id = _history_resets.get(scope.memory_scope_key, 0)
        result = await generate_reply_core(
            msg,
            current_user={"username": "nonebot", "user_id": 0},
            message_db=db,
        )
        return result

    result = await inference_runtime.submit(
        _queued_generation,
        session_id=json.dumps(scope.memory_scope_key, separators=(",", ":")),
        priority=inference_runtime.priority_for("nonebot", conversation_type),
        timeout=read_positive_float(os.environ, "MODEL_INFERENCE_TIMEOUT", 180.0),
    )
    reply = result.reply

    cost_time = round(time.time() - start_time, 2)
    logger.info(f"准备回复: 字符数={len(reply)}, cost={cost_time}s")

    # 动态延迟
    import random
    base_delay = config.REPLY_DELAY
    type_time = min(len(reply) * 0.06, 4.0)
    jitter = random.uniform(-0.3, 0.3)
    await asyncio.sleep(max(base_delay + type_time + jitter, 0.2))

    return reply


async def handle_claw(bot: Bot, user_message: str, event: MessageEvent, lora_name: str) -> str:
    """处理工具命令"""
    char_name = _get_char_name(lora_name)
    session_id = str(event.user_id)

    tool_list = "\n".join(
        [f"{name}: {tool['description']}" for name, tool in TOOLS.items()]
    )
    prompt = f"""
你是一个命令解释器。根据用户请求，选择最合适的工具并输出JSON。
[可用工具]
{tool_list}

[用户请求]
{user_message}

[输出要求]
只输出JSON，格式: {{"tool": "工具名", "args": {{"参数名": "值"}}}}
如果没有参数，args设为{{}}。"""
    json_reply = await _unified_generate(prompt, session_id, lora_name)
    logger.info(f"[claw] LLM输出: 字符数={len(json_reply)}")

    try:
        call = ToolCall.model_validate_json(json_reply)
    except ValidationError as exc:
        raise ToolError("模型返回的工具调用格式无效，本次未执行工具。") from exc
    raw = await execute_tool(call.tool, call.args, bot=bot, event=event)
    if call.tool == "write_code":
        await bot.send(event, f" **{call.args['filename']}**：\n```python\n{call.args['code']}\n```")
        await bot.send(event, raw)
    done = await _unified_generate(
        f"以{char_name}的语气如实报告工具结果。请求:{user_message}。实际结果:{raw}", session_id, lora_name,
    )
    await bot.send(event, done)


# ============================================
# 启动入口
# ============================================
def init_bot():
    """初始化机器人"""
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

    logger.info("=" * 50)
    logger.info(f"QQ自动回复机器人 - 当前选择: {_selected_lora or '跟随后端配置'}")
    logger.info("=" * 50)
    logger.info("模型推理与检索: 统一后端生成服务")
    logger.info(f"回复延迟: {config.REPLY_DELAY}秒")
    logger.info(f"数据库: 通过db.adapter统一访问")
    logger.info("=" * 50)

    # 初始化NoneBot
    nonebot.init(driver="~fastapi")

    driver = nonebot.get_driver()
    driver.register_adapter(OneBotV11Adapter)

    # ============================================
    # 消息处理器
    # ============================================
    message_handler = on_message(priority=10, block=True)

    @message_handler.handle()
    async def handle_all_messages(bot: Bot, event: MessageEvent):
        """处理所有消息"""
        logger.info("消息处理器被触发")

        try:
            if not await should_reply(event):
                logger.info("不满足回复条件，跳过")
                return
            msg_id = str(event.message_id)
            if await _is_duplicate_message(msg_id):
                logger.debug("跳过重复消息")
                return

            user_message = str(event.message).strip()
            logger.info(f"[路由] user_id={event.user_id}, 是否claw={claw_sessions.get(str(event.user_id), False)}, 字符数={len(user_message)}")

            if not claw_sessions.get(str(event.user_id), False):
                if user_message == "/claw":
                    if not isinstance(event, PrivateMessageEvent):
                        await message_handler.finish("请在私聊中使用")
                        return
                    if is_superuser(event):
                        claw_sessions[str(event.user_id)] = True
                        await message_handler.finish("你已进入claw模式")
                    else:
                        await message_handler.finish("你不是超级用户，不能使用该命令")
                        return
                else:
                    logger.info(f"[聊天] 普通消息，走{_selected_lora}回复")

                    # 分析型问题偶尔发预消息
                    import random
                    think_keywords = ["为什么", "怎么", "分析", "解释", "说说", "讲讲", "如何"]
                    is_analytical = any(kw in user_message for kw in think_keywords)
                    if is_analytical and len(user_message) > 10 and random.random() < 0.35:
                        try:
                            pre_msgs = ["让我想想哈...", "嗯...我想想~", "这个问题有意思，等我想想~"]
                            await bot.send(event, random.choice(pre_msgs))
                        except Exception:
                            pass

                    reply = await process_message(event)
                    logger.info(f"准备发送回复: 字符数={len(reply)}")

                    # 多消息分段
                    import random

                    def _strip_outer_brackets(text: str) -> str:
                        WRAP_PAIRS = [('「', '」'), ('『', '』'), ('"', '"'), ("'", "'")]
                        stripped = text.strip()
                        changed = True
                        while changed:
                            changed = False
                            for left, right in WRAP_PAIRS:
                                if stripped.startswith(left) and stripped.endswith(right):
                                    inner = stripped[len(left):-len(right)]
                                    lc_inner = inner.count(left)
                                    rc_inner = inner.count(right)
                                    if lc_inner == rc_inner:
                                        stripped = inner.strip()
                                        changed = True
                        return stripped

                    def _clean_fragment(text: str) -> str:
                        s = text.strip()
                        if s.startswith('」') and '「' not in s[:s.index('」')]:
                            s = s[1:].strip()
                        if s.endswith('「') and '」' not in s[s.rindex('「'):]:
                            s = s[:-1].strip()
                        if s.startswith('』') and '『' not in s[:s.index('』')]:
                            s = s[1:].strip()
                        if s.endswith('『') and '』' not in s[s.rindex('『'):]:
                            s = s[:-1].strip()
                        for ch in ('"', '"'):
                            if s.startswith(ch) and s.count(ch) % 2 == 1:
                                s = s[1:].strip()
                            if s.endswith(ch) and s.count(ch) % 2 == 1:
                                s = s[:-1].strip()
                        return s

                    def _smart_split_reply(text: str) -> list[str]:
                        if not text or len(text) <= 40:
                            return [text] if text else []

                        cleaned = _strip_outer_brackets(text)
                        PAIRS = [('「', '」'), ('『', '』'), ('【', '】'), ('《', '》'), ('（', '）'), ('"', '"'), ("'", "'")]

                        def _inside_pair(pos: int) -> bool:
                            for left, right in PAIRS:
                                lc = cleaned[:pos].count(left)
                                rc = cleaned[:pos].count(right)
                                if lc > rc:
                                    return True
                            return False

                        SENTENCE_ENDS = set('。！？!?\n')
                        COMMA_PAUSES = set('，,；;…—')

                        split_positions = []
                        i = 0
                        while i < len(cleaned):
                            ch = cleaned[i]
                            if ch in SENTENCE_ENDS and not _inside_pair(i):
                                split_positions.append(i + 1)
                            elif ch in COMMA_PAUSES and not _inside_pair(i):
                                chunk_len = i + 1 - (split_positions[-1] if split_positions else 0)
                                if chunk_len >= 30:
                                    split_positions.append(i + 1)
                            i += 1

                        if not split_positions:
                            return [cleaned]

                        parts = []
                        prev = 0
                        for pos in split_positions:
                            segment = cleaned[prev:pos].strip()
                            if segment:
                                parts.append(segment)
                            prev = pos
                        if prev < len(cleaned):
                            tail = cleaned[prev:].strip()
                            if tail:
                                parts.append(tail)

                        merged = []
                        buf = parts[0] if parts else ""
                        for seg in parts[1:]:
                            if len(buf) < 20:
                                buf += seg
                            else:
                                merged.append(_clean_fragment(buf))
                                buf = seg
                        if buf:
                            merged.append(_clean_fragment(buf))

                        return merged if len(merged) > 1 else [cleaned]

                    parts = _smart_split_reply(reply)
                    if len(parts) > 1 and len(reply) > 40:
                        first = parts[0]
                        rest = "".join(parts[1:])
                        await bot.send(event, first)
                        gap = random.uniform(0.4, 1.0)
                        logger.info(f"分段发送: 第1条已发, {gap:.1f}s后发剩余")
                        await asyncio.sleep(gap)
                        await bot.send(event, rest)
                        await message_handler.finish()
                    else:
                        await message_handler.finish(Message(reply))
                    logger.info("回复发送成功")
            else:
                logger.info(f"[claw] 进入操作模式处理: 字符数={len(user_message)}")
                if user_message == "/exit":
                    del claw_sessions[str(event.user_id)]
                    await message_handler.finish("你已退出claw模式")
                    return
                else:
                    logger.info("开始处理消息...")
                    await handle_claw(bot, user_message, event, _selected_lora)
                    logger.info("回复发送成功")
        except FinishedException:
            logger.info("消息处理完成")
        except MessageDeduplicationError as exc:
            logger.error(f"消息去重失败: {type(exc.__cause__ or exc).__name__}")
            await message_handler.finish("消息去重状态无法确认，本轮未生成回复。请检查 Redis 服务后重新发送消息。")
        except SessionReplyCheckError as exc:
            logger.error(f"读取会话回复设置失败: {type(exc.__cause__ or exc).__name__}")
            await message_handler.finish("会话回复设置读取失败，本轮未生成回复。请检查数据库服务后重新发送消息。")
        except RateLimitExceeded:
            logger.warning("消息触发限流")
            await message_handler.finish("请求过于频繁，请稍后再试。")
        except InferenceQueueFull:
            logger.warning("推理队列已满")
            await message_handler.finish("[系统提示] 推理队列已满，请稍后再试。")
        except asyncio.TimeoutError:
            logger.warning("消息处理超时")
            await message_handler.finish("[系统提示] 消息处理超时，本轮回复未完成，请稍后重试。")
        except ToolError as exc:
            logger.error("工具调用未完成: {}", type(exc.__cause__ or exc).__name__)
            await message_handler.finish(f"[系统提示] {exc}")
        except Exception as exc:
            import traceback

            logger.error("消息处理失败: {}\n{}", type(exc).__name__, "".join(traceback.format_tb(exc.__traceback__)))
            await message_handler.finish("[系统提示] 消息处理失败，本轮回复未完成。请检查后端服务后重新发送消息。")

    # ============================================
    # 命令处理器
    # ============================================
    help_cmd = on_command("help", aliases={"帮助"}, priority=5)

    @help_cmd.handle()
    async def handle_help(bot: Bot, event: MessageEvent):
        """帮助命令"""
        char_name = _get_char_name(_selected_lora)
        help_text = f"""{char_name} - QQ自动回复机器人

使用说明：
- 私聊直接对话即可获得回复
- 群聊中请@{config.BOT_NAME} 或提到我的名字
- 机器人会记住最近的对话历史

命令：
/help 或 /帮助 - 查看帮助
/clear 或 /清除 - 清除对话历史
/lora - 查看可用角色
/lora <登记名称或default> - 超管切换后续消息使用的模型

当前选择: {_selected_lora or '跟随后端配置'}"""

        await help_cmd.finish(help_text)

    clear_cmd = on_command("clear", aliases={"清除"}, priority=5)

    @clear_cmd.handle()
    async def handle_clear(bot: Bot, event: MessageEvent):
        """Reset only the caller's context after earlier queued turns finish."""
        scope = _message_scope(event)

        async def clear_context():
            _history_resets[scope.memory_scope_key] = await asyncio.to_thread(db.get_message_high_watermark)

        await inference_runtime.submit(
            clear_context,
            session_id=json.dumps(scope.memory_scope_key, separators=(",", ":")),
            priority=inference_runtime.priority_for("nonebot", scope.conversation_type),
            timeout=read_positive_float(os.environ, "MODEL_INFERENCE_TIMEOUT", 180.0),
        )
        await clear_cmd.finish("当前对话上下文已重置，聊天存档和长期记忆仍保留。")

    # LoRA切换命令
    lora_cmd = on_command("lora", priority=5)

    @lora_cmd.handle()
    async def handle_lora(bot: Bot, event: MessageEvent):
        """Display registered adapters or change the bot-wide selection."""
        args = str(event.message).strip().split()
        if len(args) > 1 and not is_superuser(event):
            await lora_cmd.finish("仅超级用户可以更改机器人全局模型选择。")
            return
        try:
            if len(args) < 2:
                loras = await asyncio.to_thread(db.get_loras)
                names = ", ".join(["default", *[item["name"] for item in loras]])
                reply = f"已登记: {names}\n当前: {_selected_lora or '跟随后端配置'}\n用法: /lora <登记名称或default>"
            elif len(args) == 2:
                selected = await _select_lora(args[1])
                reply = f"已选择 {selected}，后续消息由统一后端使用该模型生成。"
            else:
                reply = "用法: /lora <登记名称或default>"
        except LoraSelectionError as exc:
            reply = f"切换失败，选择未更改：{exc}"
        except Exception as exc:
            logger.error("LoRA 选择失败: {}", type(exc).__name__)
            reply = "LoRA 读取或选择失败，选择未更改。请检查数据库及模型服务配置。"
        await lora_cmd.finish(reply)

    logger.info("机器人初始化完成")
    bot_port = int(os.getenv("BOT_PORT", "8081"))
    logger.info("提示: 请配置NapCat连接到 ws://服务器IP:%s/onebot/v11/ws", bot_port)
    logger.info("提示: 使用 /help 或 /帮助 查看帮助")

    nonebot.run(host="0.0.0.0", port=bot_port)


if __name__ == "__main__":
    init_bot()
