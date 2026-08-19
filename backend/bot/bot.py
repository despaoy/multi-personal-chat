"""
QQ自动回复机器人 - 多LoRA角色风格
基于NoneBot2 + vLLM + LoRA热切换
支持私聊和@消息识别

集成架构：
  Bot → db.adapter（统一数据库访问）→ SQLite/PostgreSQL
  Bot → HTTP API（RAG检索等需要后端进程状态的服务）→ FastAPI后端
"""

import os
import sys
import json
import time
import asyncio
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

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
from bot.tools import execute_tool, get_tools, TOOLS

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

# 统一生成链路：NoneBot 不再自行组装 prompt/RAG/失败回退
from api.generate import generate_reply_core
from db.schemas import MessageRequest
from infra.concurrency_control import (
    InferenceQueueFull,
    RateLimitExceeded,
    inference_runtime,
)

# ============================================
# 消息去重（幂等设计）
# ============================================
_processed_messages: Dict[str, float] = {}  # {message_id: timestamp}
_DEDUP_TTL = 3600  # 1小时TTL
_DEDUP_MAX_SIZE = 10000  # 内存去重集合最大容量

# ============================================
# 统一生成链路下 NoneBot 不再维护自己的 RAG/Ollama HTTP 客户端
# ============================================
async def _close_bot_http_clients() -> None:
    """保留兼容入口；统一生成服务不依赖本模块自有 HTTP 客户端。"""
    pass

async def _is_duplicate_message(message_id: str) -> bool:
    """检查消息是否已处理（幂等去重）"""
    # 尝试Redis去重（使用统一的 async 客户端）
    try:
        from cache.redis_client import get_async_redis
        redis = await get_async_redis()
        if redis:
            key = f"dedup:msg:{message_id}"
            was_set = await redis.set(key, "1", nx=True, ex=_DEDUP_TTL)
            return was_set is None or was_set is False
    except Exception as exc:
        logger.debug("Redis 去重失败，回退到内存去重: %s", exc)

    # 回退到内存去重
    import time
    now = time.time()
    if len(_processed_messages) > _DEDUP_MAX_SIZE:
        expired = [k for k, v in _processed_messages.items() if now - v > _DEDUP_TTL]
        for k in expired:
            del _processed_messages[k]

    if message_id in _processed_messages:
        return True

    _processed_messages[message_id] = now
    return False

# ============================================
# 配置 - 通过db.adapter读取
# ============================================

def _load_db_config() -> Dict[str, Any]:
    """从数据库加载配置，返回字典（通过db.adapter统一访问）"""
    try:
        return db.config
    except Exception as exc:
        logger.warning("加载数据库配置失败，返回空配置: %s", exc)
        return {}

# DB 配置缓存统一委托给 cache.config_cache（60s TTL + jitter + Redis 共享），
# 消除三套独立缓存导致的状态不同步问题。
# 此前 bot 维护独立 30s 缓存，与 config_cache 的 60s 缓存失效不联动，
# 导致配置更新后 bot 层与 API 层行为不一致。
def _get_db_config() -> Dict[str, Any]:
    """获取数据库配置（统一走 cache.config_cache，设置页修改后 60s 内全局生效）"""
    try:
        from cache.config_cache import get_cached_config, set_cached_config
        cached = get_cached_config()
        if cached is not None:
            return dict(cached)
    except Exception:
        pass

    config = _load_db_config()
    try:
        from cache.config_cache import set_cached_config
        set_cached_config(config)
    except Exception:
        pass
    return config

class Config:
    """机器人配置 - 环境变量项为静态属性，数据库项通过@property动态读取（30秒缓存）"""
    # 环境变量配置（静态，启动时读取一次）
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    SUPERUSERS = [str(u) for u in json.loads(os.getenv("SUPERUSERS", "[]"))]
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

    # LoRA风格配置
    USE_LORA_STYLE = True

    # 以下配置通过 @property 动态读取 db.config，设置页修改后30秒内自动生效
    @property
    def BOT_NAME(self):
        return _get_db_config().get('botName', os.getenv("NICKNAME", "胡桃"))

    @property
    def REPLY_DELAY(self):
        return float(_get_db_config().get('replyDelay', os.getenv("REPLY_DELAY", "0.8")))

    @property
    def AUTO_REPLY(self):
        return _get_db_config().get('autoReply', True)

    @property
    def GROUP_REPLY(self):
        return _get_db_config().get('groupReply', True)

    @property
    def PRIVATE_REPLY(self):
        return _get_db_config().get('privateReply', True)

    @property
    def TEMPERATURE(self):
        return float(_get_db_config().get('temperature', 0.7))

    @property
    def MAX_TOKENS(self):
        return int(_get_db_config().get('maxTokens', 2048))

    @property
    def CONTEXT_WINDOW(self):
        return _get_db_config().get('contextWindow', '8k')

    @property
    def USE_KNOWLEDGE_BASE(self):
        return _get_db_config().get('useKnowledgeBase', True)

    @property
    def CONTENT_FILTER(self):
        return _get_db_config().get('contentFilter', True)

    @property
    def CONTENT_REVIEW(self):
        return _get_db_config().get('contentReview', True)

    @property
    def ADMIN_QQ_LIST(self):
        return _get_db_config().get('adminQqList', '')

    @property
    def ERROR_ALERT(self):
        return _get_db_config().get('errorAlert', True)

    @property
    def DAILY_STATS(self):
        return _get_db_config().get('dailyStats', True)

    @property
    def ANOMALY_DETECTION(self):
        return _get_db_config().get('anomalyDetection', False)

    @property
    def DEFAULT_REPLY_TEMPLATE(self):
        return _get_db_config().get('defaultReplyTemplate', '')

config = Config()

# ============================================
# 会话历史管理 - 通过db.adapter读取
# ============================================
class SessionHistory:
    """会话历史管理器 - 通过db.adapter恢复历史，内存中管理

    C14 fix: 用 threading.Lock 保护 self.sessions 的 check-then-set 序列，
    防止 async 协程在 await 点交错时同一 session_id 的历史列表损坏或消息丢失。
    单个方法内无 await，GIL 下同步方法是原子的，但跨方法调用（如 get_history 后 add_message）
    需要锁保护以避免其他协程在中间修改。
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self._lock = threading.Lock()

    def _load_from_db(self, session_id: str):
        """从数据库恢复最近N轮对话（通过db.adapter）"""
        try:
            # 修复：使用 SQL 层 session_id 过滤，避免取全局最近 10 条后 Python 过滤
            #   原实现 db.get_messages(limit=10) 取全局前 10 条 → 若当前会话不在前 10 则颗粒无收
            messages = db.get_messages(limit=10, session_id=session_id)
            # 按时间正序排列（SQL 返回 DESC，reversed 后为 ASC——最早→最新）
            session_msgs = list(reversed(messages))

            result = []
            for msg in session_msgs:
                result.append({"role": "user", "content": msg.get("message", "")})
                result.append({"role": "assistant", "content": msg.get("reply", "")})

            if result:
                self.sessions[session_id] = result
                logger.info(f"会话 {session_id} 从数据库恢复 {len(result)//2} 轮历史")
        except Exception as e:
            logger.debug(f"恢复会话 {session_id} 失败（可能首次对话）: {e}")

    def get_history(self, session_id: str, tokenizer=None) -> List[Dict[str, str]]:
        """获取会话历史，按token数截断"""
        with self._lock:
            if session_id not in self.sessions:
                self._load_from_db(session_id)
                if session_id not in self.sessions:
                    return []

            history = self.sessions[session_id]
            trimmed = []
            total = 0
            for msg in reversed(history):
                tokens = self._count_tokens(msg["content"], tokenizer)
                if total + tokens > self.max_tokens:
                    break
                trimmed.insert(0, msg)
                total += tokens
            return trimmed

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到内存历史"""
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append({"role": role, "content": content})
            self._prune_by_tokens(session_id)

    def _prune_by_tokens(self, session_id: str):
        """按token数裁剪历史"""
        history = self.sessions.get(session_id, [])
        total = 0
        for msg in reversed(history):
            total += self._count_tokens(msg["content"], None)
        while total > self.max_tokens * 2 and len(history) > 2:
            removed = history.pop(0)
            total -= self._count_tokens(removed["content"], None)

    def clear_history(self, session_id: str):
        """清除会话历史"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

    @staticmethod
    def _count_tokens(text: str, tokenizer=None) -> int:
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 2)

# 全局会话历史管理器
_context_window_map = {'4k': 4000, '8k': 8000, '16k': 16000, '32k': 32000}
_context_tokens = _context_window_map.get(str(config.CONTEXT_WINDOW), 2000)
session_history = SessionHistory(max_tokens=_context_tokens // 2)
claw_sessions: Dict[str, bool] = {}

# ============================================
# 多LoRA热切换模型管理
# ============================================
_hutao_7b_model = None
_hutao_7b_tokenizer = None
# C12 fix: 保护 _hutao_7b_model/_hutao_7b_tokenizer/_current_lora 的 check-then-set 序列。
# 防止消息推理过程中 /lora 切换命令并发修改模型，导致正在进行的推理使用错误 LoRA。
_lora_model_lock: asyncio.Lock | None = None
_lora_model_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_lora_model_lock() -> asyncio.Lock:
    """Return the model mutation lock for the active bot event loop."""
    global _lora_model_lock, _lora_model_lock_loop
    loop = asyncio.get_running_loop()
    if _lora_model_lock is None or _lora_model_lock_loop is not loop:
        _lora_model_lock = asyncio.Lock()
        _lora_model_lock_loop = loop
    return _lora_model_lock


def _get_active_lora_from_db() -> str:
    """从后端数据库读取当前激活的LoRA名称（通过db.adapter统一访问）"""
    try:
        loras = db.get_loras(status="active")
        if loras:
            name = loras[0].get('name', '')
            if name in LORA_REGISTRY:
                return name
            # 前缀匹配（数据库中可能是hutao_lora_7b，LORA_REGISTRY中是hutao）
            for key in LORA_REGISTRY:
                if name.startswith(key) or key.startswith(name.split("_")[0]):
                    return key
    except Exception as e:
        logger.warning(f"从db adapter读取LoRA状态失败: {e}")
    return _current_lora


def _sync_current_lora():
    """从数据库同步当前LoRA到内存变量"""
    global _current_lora
    active = _get_active_lora_from_db()
    logger.info(f"LoRA同步检查: DB返回={active}, 当前={_current_lora}")
    if active != _current_lora:
        logger.info(f"LoRA 同步: {_current_lora} → {active}")
        _current_lora = active


_current_lora = "hutao"

_BACKEND_DIR = Path(__file__).parent
_BACKEND_ROOT = _BACKEND_DIR.parent  # backend/ 目录


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return str(_BACKEND_ROOT / p)


# H1 fix: LORA_REGISTRY 已抽取到 inference/lora_registry.py 作为中立层，
# 消除 api/generate.py 对 bot 层的反向依赖。bot.py 保留导入以兼容现有代码。
from inference.lora_registry import LORA_REGISTRY, LORA_NAMES, get_lora_system_prompt


def _get_char_name(lora_name: str = None) -> str:
    """从 LORA_REGISTRY 中提取角色名称"""
    from inference.lora_registry import get_char_name as _get_char_name_impl
    return _get_char_name_impl(lora_name, _current_lora)


def _load_7b_model(lora_name: str = None):
    """加载 Qwen3-8B 4bit + 指定 LoRA 适配器（支持热切换）"""
    global _hutao_7b_model, _hutao_7b_tokenizer, _current_lora

    lora_name = lora_name or _current_lora
    if lora_name not in LORA_REGISTRY:
        logger.warning(f"未知 LoRA: {lora_name}，回退到 hutao")
        lora_name = "hutao"

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    # 首次加载
    if _hutao_7b_model is None:
        base_model_path = os.getenv("BASE_MODEL_PATH", "models/Qwen3-8B-Instruct")
        if not os.path.isabs(base_model_path):
            base_model_path = str(Path(__file__).parent / base_model_path)
        base_model_path = str(Path(base_model_path).resolve())
        if not Path(base_model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {base_model_path}，请设置 BASE_MODEL_PATH 环境变量或下载模型")

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        logger.info(f"加载 Qwen3-8B (4bit)...")
        _hutao_7b_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=nf4_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        _hutao_7b_model = PeftModel.from_pretrained(
            base_model, LORA_REGISTRY[lora_name]["path"], adapter_name=lora_name
        )
        _hutao_7b_model.eval()
        _current_lora = lora_name

        vram = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"7B 模型加载完成 (LoRA={lora_name})，显存: {vram:.1f}GB")

    # 热切换
    elif lora_name != _current_lora:
        if lora_name not in _hutao_7b_model.peft_config:
            logger.info(f"加载新 LoRA 适配器: {lora_name}")
            _hutao_7b_model.load_adapter(
                LORA_REGISTRY[lora_name]["path"], adapter_name=lora_name
            )
        _hutao_7b_model.set_adapter(lora_name)
        logger.info(f"LoRA 切换: {_current_lora} → {lora_name}")
        _current_lora = lora_name

    return _hutao_7b_model, _hutao_7b_tokenizer


def is_superuser(event: MessageEvent) -> bool:
    user_id = str(event.user_id)
    return user_id in config.SUPERUSERS


async def _unified_generate(prompt: str, session_id: str, lora_name: str | None = None) -> str:
    """Run any NoneBot-side generation through the shared ChatGenerationService."""
    msg = MessageRequest(
        message=prompt,
        sessionType="private",
        conversationType="private",
        sessionId=session_id or "claw",
        sessionName="nonebot",
        userId="nonebot",
        userName="nonebot",
        senderName="nonebot",
        platform="qq",
        adapter="nonebot",
        conversationId=session_id or "claw",
        senderId="nonebot",
        loraName=lora_name or "",
        traceId="",
    )
    result = await inference_runtime.submit(
        lambda: generate_reply_core(
            msg,
            current_user={"username": "nonebot", "user_id": 0},
            persist_message=False,
            enable_rag=False,
            record_invocation=False,
        ),
        session_id=session_id or "claw",
        priority=inference_runtime.priority_for("nonebot", "private"),
        timeout=float(os.getenv("MODEL_INFERENCE_TIMEOUT", "180")),
    )
    return result.reply


async def generate_with_local_model(prompt: str, session_id: Optional[str] = None, is_claw: bool = False, lora_name: str = None) -> str:
    """Legacy-compatible wrapper: all generation now uses the shared service."""
    return await _unified_generate(prompt, session_id or "claw", lora_name or _current_lora)


# ============================================
# 消息处理
# ============================================
async def save_message_to_backend(event: MessageEvent, message: str, reply: str, cost_time: float):
    """保存消息到后端数据库（通过db.adapter统一访问）"""
    try:
        session_type = "private" if isinstance(event, PrivateMessageEvent) else "group"
        session_id = str(event.user_id) if isinstance(event, PrivateMessageEvent) else str(event.group_id)
        lora_name = _current_lora or "default"

        message_data = {
            "sessionType": session_type,
            "sessionId": session_id,
            "sessionName": str(event.sender.nickname) if hasattr(event.sender, 'nickname') else "未知用户",
            "userId": str(event.user_id),
            "userName": str(event.sender.nickname) if hasattr(event.sender, 'nickname') else "未知用户",
            "message": message,
            "reply": reply,
            "modelName": config.OLLAMA_MODEL,
            "loraName": lora_name,
            "costTime": cost_time,
            "createdAt": datetime.now().isoformat()
        }

        db.add_message(message_data)
        logger.info("消息已保存到数据库")

    except Exception as e:
        logger.warning(f"保存消息到数据库失败: {e}")

async def should_reply(event: MessageEvent) -> bool:
    """判断是否应该回复该消息"""
    # 检查该会话是否启用了机器人（通过db.adapter）
    try:
        is_group = isinstance(event, GroupMessageEvent)
        session_id = str(event.group_id) if is_group else str(event.user_id)
        conversation_type = "group" if is_group else "private"
        enabled = await asyncio.to_thread(
            db.is_session_bot_enabled,
            session_id,
            "qq",
            session_id,
            conversation_type,
        )
        if not enabled:
            logger.info(f"会话 {session_id} 机器人已关闭，跳过")
            return False
    except Exception:
        pass

    # 私聊消息总是回复
    if isinstance(event, PrivateMessageEvent):
        logger.info("私聊消息，直接回复")
        return True

    # 群聊消息
    if isinstance(event, GroupMessageEvent):
        message_text = str(event.message)
        # M5 fix: 群聊消息可能含 PII（用户姓名、手机号、私聊内容等），
        # 此前直接 logger.info 全文会写入日志文件造成泄露。改为只记长度与短预览。
        logger.info(f"群聊消息: len={len(message_text)}, preview={message_text[:30]!r}")

        if event.is_tome():
            logger.info("检测到@机器人，回复")
            return True
        if config.BOT_NAME in message_text:
            logger.info("检测到包含机器人名称，回复")
            return True
        trigger_words = ["你好", "在吗", "有人吗", config.BOT_NAME, "@ad"]
        char_name = _get_char_name(_current_lora)
        if char_name != config.BOT_NAME:
            trigger_words.append(char_name)
        for word in trigger_words:
            if word in message_text:
                logger.info(f"检测到触发词'{word}'，回复")
                return True

        logger.info("不满足回复条件")

    return False


async def process_message(event: MessageEvent) -> str:
    """处理消息并生成回复。

    NoneBot 只负责消息接收/回复条件/平台适配；模型生成统一走
    ChatGenerationService + InferenceRuntime，与 API/AstrBot 同一条链路。
    """
    user_message = str(event.message).strip()

    if not user_message:
        return "嗯？怎么不说话呀？"

    logger.info(f"收到消息: len={len(user_message)}, preview={user_message[:30]!r}")

    # C17 fix: 统一群消息 session_id 为 str(group_id)，与 should_reply 和
    # save_message_to_backend 保持一致。
    is_private = isinstance(event, PrivateMessageEvent)
    conversation_id = str(event.user_id) if is_private else str(event.group_id)
    session_id = conversation_id
    conversation_type = "private" if is_private else "group"
    sender_id = str(event.user_id)
    sender_obj = getattr(event, "sender", None)
    sender_name = str(
        getattr(sender_obj, "card", None)
        or getattr(sender_obj, "nickname", None)
        or event.user_id
    )

    start_time = time.time()
    _sync_current_lora()
    logger.info(f"开始处理消息: {user_message[:50]}... (LoRA={_current_lora})")

    try:
        await inference_runtime.check_rate_limits("qq", conversation_id, sender_id)
    except RateLimitExceeded:
        logger.warning("NoneBot 消息触发限流 conversation=%s", conversation_id)
        reply = "请求过于频繁，请稍后再试。"
    else:
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
            traceId="",
        )

        async def _queued_generation():
            # Read and update history while InferenceRuntime still holds the
            # session lock, so two fast messages cannot both snapshot old
            # history and lose the first reply.
            msg.history = session_history.get_history(session_id)
            result = await generate_reply_core(
                msg,
                current_user={"username": "nonebot", "user_id": 0},
            )
            session_history.add_message(session_id, "user", user_message)
            session_history.add_message(session_id, "assistant", result.reply)
            return result

        try:
            result = await inference_runtime.submit(
                _queued_generation,
                session_id=session_id,
                priority=inference_runtime.priority_for("nonebot", conversation_type),
                timeout=float(os.getenv("MODEL_INFERENCE_TIMEOUT", "180")),
            )
            reply = result.reply
        except InferenceQueueFull:
            logger.warning("NoneBot 推理队列已满 conversation=%s", conversation_id)
            reply = "[系统提示] 当前消息较多，请稍后再试。"
        except asyncio.TimeoutError:
            logger.warning("NoneBot 推理排队超时 conversation=%s", conversation_id)
            reply = "[系统提示] 当前处理较慢，请稍后再试。"
        except Exception as exc:
            logger.error(f"NoneBot 统一生成失败: {type(exc).__name__}: {exc}", exc_info=True)
            reply = "[系统提示] AI 推理服务暂不可用，请稍后再试或联系管理员"

    cost_time = round(time.time() - start_time, 2)
    logger.info(f"发送回复: len={len(reply)}, cost={cost_time}s, preview={reply[:30]!r}")

    # 动态延迟
    import random
    base_delay = config.REPLY_DELAY
    type_time = min(len(reply) * 0.06, 4.0)
    jitter = random.uniform(-0.3, 0.3)
    await asyncio.sleep(max(base_delay + type_time + jitter, 0.2))

    return reply


async def llm_raw(prompt: str) -> str:
    """Raw LLM helper now uses the same unified generation chain."""
    return await _unified_generate(prompt, "claw-raw", _current_lora)


async def call_llm_claw(prompt: str, lora_name: str = None) -> str:
    """claw 模式专用推理调用"""
    return await generate_with_local_model(prompt, is_claw=True, lora_name=lora_name)


async def handle_claw(bot: Bot, user_message: str, event: MessageEvent, lora_name: str = None) -> str:
    """处理工具命令"""
    char_name = _get_char_name(lora_name or _current_lora)

    tool_list = "\n".join(
        [f"{name}: {tool['description']}" for name, tool in TOOLS.items()]
    )
    analysis_prompt = f"你是{char_name}，可用工具:\n{tool_list}\n用户请求：{user_message}\n简要说明你打算用什么工具来解决，风格自然。"
    thinking = await call_llm_claw(analysis_prompt, lora_name or _current_lora)
    await bot.send(event, thinking)
    # M5 fix (扩展): 思考结果可能回显用户消息中的 PII，只记长度与短预览（≤60 chars）
    logger.info(f"[claw] 思考结果: len={len(thinking)}, preview={thinking[:60]!r}")

    prompt = f"""
你是一个命令解释器。根据用户请求，选择最合适的工具并输出JSON。
[可用工具]
{tool_list}

[用户请求]
{user_message}

[输出要求]
只输出JSON，格式: {{"tool": "工具名", "args": {{"参数名": "值"}}}}
如果没有参数，args设为{{}}。"""
    json_reply = await llm_raw(prompt)
    # M5 fix: LLM 原始输出可能回显用户消息中的 PII，只记长度与短预览（≤60 chars）
    logger.info(f"[claw] LLM原始输出: len={len(json_reply)}, preview={json_reply[:60]!r}")

    text = json_reply.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0] if "```" in text else text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]

    try:
        call = json.loads(text)
    except (json.JSONDecodeError, KeyError):
        await bot.send(event, "没能理解你的意思，不如换种说法？")
        return "工具调用解析失败"

    tool_name = call["tool"]
    if tool_name == "write_code":
        explain = await call_llm_claw(f"你是{char_name}，你正在帮用户编写代码，向用户简单解释代码的功能，用户要求{user_message}", lora_name or _current_lora)
        await bot.send(event, explain)
        # M5 fix (扩展): 解释结果可能回显用户请求中的 PII，只记长度与短预览（≤60 chars）
        logger.info(f"[claw] 解释结果: len={len(explain)}, preview={explain[:60]!r}")
    raw = await execute_tool(call["tool"], call.get("args", {}), bot=bot, event=event)
    args = call.get("args", {})
    code = ""
    if tool_name == "write_code":
        code = args.get("code", "")
        filename = args.get("filename", "")
        await bot.send(event=event, message=f" **{filename}**：\n```python\n{code}\n```")

    if tool_name == "write_code":
        done = await call_llm_claw(f"你是{char_name}，代码已经编写完成{code}，向用户回复代码已经编写完成，任务结束，自然收尾。", lora_name or _current_lora)
        await bot.send(event, done)
        await bot.send(event, message=f"运行结果如下：\n{raw}")
    else:
        done = await call_llm_claw(f"以{char_name}的语气报告结果。请求:{user_message}。结果:{raw}", lora_name or _current_lora)
        await bot.send(event, done)


# ============================================
# 启动入口
# ============================================
def init_bot():
    """初始化机器人"""
    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter
    from nonebot import on_command, on_message

    logger.info("=" * 50)
    logger.info(f"QQ自动回复机器人 - 当前角色: {_current_lora}")
    logger.info("=" * 50)
    logger.info(f"Ollama地址: {config.OLLAMA_BASE_URL}")
    logger.info(f"Ollama模型: {config.OLLAMA_MODEL}")
    logger.info(f"LoRA风格: {'启用' if config.USE_LORA_STYLE else '禁用'} (当前: {_current_lora})")
    logger.info(f"回复延迟: {config.REPLY_DELAY}秒")
    logger.info(f"RAG检索: 通过后端API ({config.API_BASE_URL})")
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

        # 幂等去重
        msg_id = str(getattr(event, 'message_id', ''))
        if msg_id and await _is_duplicate_message(msg_id):
            logger.debug(f"跳过重复消息: {msg_id}")
            return

        try:
            if not await should_reply(event):
                logger.info("不满足回复条件，跳过")
                return

            user_message = str(event.message).strip()
            logger.info(f"[路由] user_id={str(event.user_id)}, 是否claw={claw_sessions.get(str(event.user_id), False)}, 消息={user_message[:50]}")

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
                    logger.info(f"[聊天] 普通消息，走{_current_lora}回复")

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
                    logger.info(f"准备发送回复: {reply[:50]}...")

                    # 多消息分段
                    import re, random

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

                    def _smart_split_reply(text: str) -> List[str]:
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
                logger.info(f"[claw] 进入操作模式处理: {user_message[:50]}")
                if user_message == "/exit":
                    del claw_sessions[str(event.user_id)]
                    await message_handler.finish("你已退出claw模式")
                    return
                else:
                    logger.info("开始处理消息...")
                    await handle_claw(bot, user_message, event, _current_lora)
                    logger.info("回复发送成功")
        except FinishedException:
            logger.info("消息处理完成")
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # ============================================
    # 命令处理器
    # ============================================
    help_cmd = on_command("help", aliases={"帮助"}, priority=5)

    @help_cmd.handle()
    async def handle_help(bot: Bot, event: MessageEvent):
        """帮助命令"""
        char_name = _get_char_name(_current_lora)
        help_text = f"""{char_name} - QQ自动回复机器人

使用说明：
- 私聊直接对话即可获得回复
- 群聊中请@{config.BOT_NAME} 或提到我的名字
- 机器人会记住最近的对话历史

命令：
/help 或 /帮助 - 查看帮助
/clear 或 /清除 - 清除对话历史
/lora - 查看可用角色
/lora <角色名> - 切换到指定角色

当前角色: {_current_lora}"""

        await help_cmd.finish(help_text)

    clear_cmd = on_command("clear", aliases={"清除"}, priority=5)

    @clear_cmd.handle()
    async def handle_clear(bot: Bot, event: MessageEvent):
        """清除对话历史命令"""
        # C17 fix: 与 should_reply/process_message/save_message_to_backend 保持一致
        # 群消息使用 str(group_id)，私聊使用 str(user_id)
        # 此前使用 f"{group_id}_{user_id}" 复合格式导致清除命令失效
        session_id = str(event.user_id) if isinstance(event, PrivateMessageEvent) else str(event.group_id)
        session_history.clear_history(session_id)
        await clear_cmd.finish("好啦好啦，对话历史已经清除了~")

    # LoRA切换命令
    lora_cmd = on_command("lora", priority=5)

    @lora_cmd.handle()
    async def handle_lora(bot: Bot, event: MessageEvent):
        """LoRA角色切换命令"""
        global _current_lora
        args = str(event.message).strip().split()
        if len(args) < 2:
            names = ", ".join(LORA_NAMES)
            await lora_cmd.finish(f"可用角色: {names}\n当前: {_current_lora} ({_get_char_name(_current_lora)})\n用法: /lora <角色名>")
            return
        target = args[1].lower()
        if target not in LORA_REGISTRY:
            await lora_cmd.finish(f"没有叫 {target} 的角色哦~ 可用: {', '.join(LORA_NAMES)}")
            return
        if target == _current_lora:
            await lora_cmd.finish(f"已经是 {target} 啦！")
            return
        # C12 fix: 与 generate_with_local_model 的 transformers 路径互斥，
        # 防止切换过程中正在进行的推理读到中间状态
        async with _get_lora_model_lock():
            try:
                _load_7b_model(target)
                await lora_cmd.finish(f"好嘞，现在我是 {target} 啦！")
            except Exception as e:
                logger.error(f"LoRA切换失败: {e}")
                await lora_cmd.finish(f"切换失败: {e}")

    logger.info("机器人初始化完成")
    bot_port = int(os.getenv("BOT_PORT", "8081"))
    logger.info("提示: 请配置NapCat连接到 ws://服务器IP:%s/onebot/v11/ws", bot_port)
    logger.info("提示: 使用 /help 或 /帮助 查看帮助")

    nonebot.run(host="0.0.0.0", port=bot_port)


if __name__ == "__main__":
    init_bot()
