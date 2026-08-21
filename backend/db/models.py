"""
SQLAlchemy ORM 模型 - 数据库 schema 的单一真相源

使用 SQLAlchemy 2.0 declarative 风格（DeclarativeBase + Mapped + mapped_column）
定义全部 28 张表。本文件作为数据库 schema 的唯一权威定义，供 alembic 迁移
和应用层共享。

字段命名与类型严格对齐 backend/db/pg_database.py 中的 PostgreSQL Core 表定义，
保持原有驼峰/下划线命名风格不做改动。
"""

from typing import Optional

from sqlalchemy import (
    Integer,
    BigInteger,
    Text,
    Float,
    String,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类"""
    pass


# 便于 alembic 引用：target_metadata = models.metadata
metadata = Base.metadata


# ============================================
# 1. 用户表
# ============================================

class User(Base):
    """用户账号表，存储认证信息与角色（RBAC）"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False, server_default="user")


# ============================================
# 2. API Key 表（统一访问控制）
# ============================================

class ApiKey(Base):
    """托管 API Key，统一存主数据库（SQLite/PostgreSQL）"""
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    revoked_at: Mapped[Optional[float]] = mapped_column(Float)
    last_used_at: Mapped[Optional[float]] = mapped_column(Float)
    is_active: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    rate_limit: Mapped[Optional[int]] = mapped_column(Integer)


# ============================================
# 3. 配置表
# ============================================

class Config(Base):
    """系统配置键值对表"""
    __tablename__ = "config"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 3. 消息记录表
# ============================================

class Message(Base):
    """聊天消息记录表，存储用户消息与模型回复"""
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_session_created", "sessionId", "createdAt"),
        Index("idx_messages_platform_conversation", "platform", "conversationId", "createdAt"),
        Index("idx_messages_source_dedup", "platform", "adapter", "sourceMessageId"),
        Index("idx_messages_created_at", "createdAt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sessionType: Mapped[str] = mapped_column(Text, nullable=False)
    sessionId: Mapped[str] = mapped_column(Text, nullable=False)
    sessionName: Mapped[Optional[str]] = mapped_column(Text)
    platform: Mapped[str] = mapped_column(Text, nullable=False, server_default="qq")
    adapter: Mapped[str] = mapped_column(Text, nullable=False, server_default="nonebot")
    conversationId: Mapped[Optional[str]] = mapped_column(Text)
    conversationType: Mapped[Optional[str]] = mapped_column(Text)
    senderId: Mapped[Optional[str]] = mapped_column(Text)
    senderName: Mapped[Optional[str]] = mapped_column(Text)
    sourceMessageId: Mapped[Optional[str]] = mapped_column(Text)
    traceId: Mapped[Optional[str]] = mapped_column(Text)
    userId: Mapped[Optional[str]] = mapped_column(Text)
    userName: Mapped[Optional[str]] = mapped_column(Text)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    reply: Mapped[str] = mapped_column(Text, nullable=False)
    modelName: Mapped[Optional[str]] = mapped_column(Text)
    loraName: Mapped[Optional[str]] = mapped_column(Text)
    costTime: Mapped[Optional[float]] = mapped_column(Float)
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 4. LoRA 模型表
# ============================================

class Lora(Base):
    """LoRA 适配器模型注册表"""
    __tablename__ = "loras"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="inactive")
    style: Mapped[Optional[str]] = mapped_column(Text)
    size: Mapped[Optional[str]] = mapped_column(Text)
    trainedSteps: Mapped[Optional[int]] = mapped_column(Integer)
    totalSteps: Mapped[Optional[int]] = mapped_column(Integer)
    createdAt: Mapped[Optional[str]] = mapped_column(Text)


# ============================================
# 5. 知识库表
# ============================================

class KnowledgeBase(Base):
    """知识库注册表"""
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, server_default="")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 6. 知识库文件夹表
# ============================================

class KnowledgeFolder(Base):
    """知识库文件夹，用于组织文档层级"""
    __tablename__ = "knowledge_folders"
    __table_args__ = (
        UniqueConstraint("knowledge_base_id", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    knowledge_base_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, server_default="")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 7. 知识库文档表
# ============================================

class KnowledgeDocument(Base):
    """知识库文档内容表"""
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        Index("idx_knowledge_documents_kb_id", "knowledge_base_id"),
        Index("idx_knowledge_documents_folder_id", "folder_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False, server_default="未分类")
    knowledge_base_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL")
    )
    folder_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("knowledge_folders.id", ondelete="SET NULL")
    )
    sourceType: Mapped[str] = mapped_column(Text, nullable=False, server_default="text")
    sourceUrl: Mapped[Optional[str]] = mapped_column(Text)
    fileType: Mapped[Optional[str]] = mapped_column(Text)
    fileSize: Mapped[Optional[int]] = mapped_column(Integer)
    chunkCount: Mapped[Optional[int]] = mapped_column(Integer, server_default="0")
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)
    updatedAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 8. 知识库文档分块表
# ============================================

class KnowledgeChunk(Base):
    """知识库文档分块表，embedding 存储 Faiss 向量 ID"""
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index("idx_knowledge_chunks_documentId", "documentId"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    documentId: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunkIndex: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[int]] = mapped_column(BigInteger)
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 9. 用户数据持久化表
# ============================================

class UserData(Base):
    """用户表单数据持久化表（按页面键存储 JSON）"""
    __tablename__ = "user_data"
    __table_args__ = (
        UniqueConstraint("user_id", "page_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    page_key: Mapped[str] = mapped_column(Text, nullable=False)
    data_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 10. 保存的对话表
# ============================================

class SavedDialogue(Base):
    """已保存的角色对话数据表"""
    __tablename__ = "saved_dialogues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    character_desc: Mapped[str] = mapped_column(Text, nullable=False)
    style: Mapped[Optional[str]] = mapped_column(Text)
    dialogue_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    dialogues_json: Mapped[str] = mapped_column(Text, nullable=False)
    turn_stats: Mapped[Optional[str]] = mapped_column(Text)
    scene_stats: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 12. Claw 工具表
# ============================================

class ClawTool(Base):
    """自定义 Claw 工具表"""
    __tablename__ = "claw_tools"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    code: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 13. 集成消息去重表
# ============================================

class IntegrationMessageDedup(Base):
    """集成平台消息去重表"""
    __tablename__ = "integration_message_dedup"

    dedupKey: Mapped[str] = mapped_column(Text, primary_key=True)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    adapter: Mapped[str] = mapped_column(Text, nullable=False)
    messageId: Mapped[str] = mapped_column(Text, nullable=False)
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 14. 会话聚合表
# ============================================

class Conversation(Base):
    """会话聚合信息表（机器人开关、回复策略等）"""
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint(
            "platform",
            "conversationId",
            "conversationType",
            name="uq_conversations_platform_conversation",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    conversationId: Mapped[str] = mapped_column(Text, nullable=False)
    conversationType: Mapped[str] = mapped_column(Text, nullable=False, server_default="private")
    displayName: Mapped[Optional[str]] = mapped_column(Text)
    botEnabled: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    replyPolicy: Mapped[str] = mapped_column(Text, nullable=False, server_default="default")
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)
    updatedAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 15. 集成事件表
# ============================================

class IntegrationEvent(Base):
    """集成平台事件记录表（含事件哈希去重）"""
    __tablename__ = "integration_events"
    __table_args__ = (
        UniqueConstraint(
            "platform",
            "adapter",
            "eventHash",
            name="uq_integration_events_platform_hash",
        ),
        Index("idx_integration_events_trace", "traceId"),
        Index("idx_integration_events_platform_created", "platform", "createdAt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    adapter: Mapped[str] = mapped_column(Text, nullable=False)
    sourceMessageId: Mapped[Optional[str]] = mapped_column(Text)
    conversationId: Mapped[Optional[str]] = mapped_column(Text)
    conversationType: Mapped[Optional[str]] = mapped_column(Text)
    senderId: Mapped[Optional[str]] = mapped_column(Text)
    eventType: Mapped[str] = mapped_column(Text, nullable=False, server_default="message")
    eventHash: Mapped[str] = mapped_column(Text, nullable=False)
    rawSummary: Mapped[Optional[str]] = mapped_column(Text)
    traceId: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="received")
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 16. 模型调用记录表
# ============================================

class ModelInvocation(Base):
    """模型调用记录表（含 token 计量与错误类型）"""
    __tablename__ = "model_invocations"
    __table_args__ = (
        Index("idx_model_invocations_trace", "traceId"),
        Index("idx_model_invocations_created", "createdAt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    traceId: Mapped[Optional[str]] = mapped_column(Text)
    platform: Mapped[str] = mapped_column(Text, nullable=False, server_default="qq")
    conversationId: Mapped[Optional[str]] = mapped_column(Text)
    sessionId: Mapped[Optional[str]] = mapped_column(Text)
    modelName: Mapped[Optional[str]] = mapped_column(Text)
    loraName: Mapped[Optional[str]] = mapped_column(Text)
    costTime: Mapped[Optional[float]] = mapped_column(Float, server_default="0")
    promptTokens: Mapped[Optional[int]] = mapped_column(Integer, server_default="0")
    completionTokens: Mapped[Optional[int]] = mapped_column(Integer, server_default="0")
    totalTokens: Mapped[Optional[int]] = mapped_column(Integer, server_default="0")
    usedRag: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    usedLora: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    errorType: Mapped[Optional[str]] = mapped_column(Text, server_default="")
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 17. 审计日志表
# ============================================

class AuditLog(Base):
    """API 审计日志表"""
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("idx_audit_logs_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    resource: Mapped[Optional[str]] = mapped_column(Text)
    detail: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(Text)


# ============================================
# 18. 意图样本表
# ============================================

class IntentSample(Base):
    """意图分类训练样本表"""
    __tablename__ = "intent_samples"
    __table_args__ = (
        Index("idx_intent_samples_kbName", "kbName"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kbName: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    createdAt: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 19. 意图活跃知识库表
# ============================================

class IntentActiveKb(Base):
    """意图路由启用的知识库列表"""
    __tablename__ = "intent_active_kbs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kbName: Mapped[str] = mapped_column(Text, nullable=False)
    isActive: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")


# ============================================
# 20. 训练任务表
# ============================================

class TrainingTask(Base):
    """LoRA/训练任务记录表。

    Existing PostgreSQL deployments may still contain ignored legacy columns,
    but the active schema is the same compact contract used by SQLite.
    """
    __tablename__ = "training_tasks"
    __table_args__ = (
        Index("idx_training_tasks_task_id", "task_id", unique=True),
        Index("idx_training_tasks_lora_name", "lora_name"),
        Index("idx_training_tasks_status", "status"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    task_id: Mapped[Optional[str]] = mapped_column(Text)
    lora_name: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    progress: Mapped[Optional[float]] = mapped_column(Float, server_default="0")
    error_message: Mapped[Optional[str]] = mapped_column(Text, server_default="")
    config_json: Mapped[Optional[str]] = mapped_column(Text, server_default="{}")
    created_at: Mapped[Optional[str]] = mapped_column(Text, server_default="")
    updated_at: Mapped[Optional[str]] = mapped_column(Text, server_default="")


# ============================================
# 21. Gold 评估运行表
# ============================================

class GoldEvalRun(Base):
    """Gold 评估集运行结果表"""
    __tablename__ = "gold_eval_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_at: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_name: Mapped[Optional[str]] = mapped_column(Text)
    model_label: Mapped[Optional[str]] = mapped_column(Text)
    total_prompts: Mapped[Optional[int]] = mapped_column(Integer, server_default="0")
    category_breakdown: Mapped[Optional[str]] = mapped_column(Text)
    metrics: Mapped[Optional[str]] = mapped_column(Text)
    config_snapshot: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)


# ============================================
# 22. 实验运行表
# ============================================

class ExperimentRun(Base):
    """实验运行记录表"""
    __tablename__ = "experiment_runs"
    __table_args__ = (
        Index("idx_experiment_runs_type", "experiment_type"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    experiment_type: Mapped[str] = mapped_column(Text, nullable=False)
    hypothesis: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    started_at: Mapped[str] = mapped_column(Text, nullable=False)
    completed_at: Mapped[Optional[str]] = mapped_column(Text)
    results: Mapped[Optional[str]] = mapped_column(Text)
    config_path: Mapped[Optional[str]] = mapped_column(Text)
    report_path: Mapped[Optional[str]] = mapped_column(Text)


# ============================================
# 23. 检索评估问题表
# ============================================

class RetrievalEvalQuestion(Base):
    """检索评估问题集表"""
    __tablename__ = "retrieval_eval_questions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_doc_ids: Mapped[Optional[str]] = mapped_column(Text)
    expected_doc_titles: Mapped[Optional[str]] = mapped_column(Text)
    gold_answer: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 24. 偏好对表
# ============================================

class PreferencePair(Base):
    """DPO/偏好对齐训练数据表。

    注：Python 属性名使用 metadata_ 以避免与 DeclarativeBase.metadata 冲突，
    数据库列名仍为 "metadata"。
    """
    __tablename__ = "preference_pairs"
    __table_args__ = (
        Index("idx_preference_pairs_status", "review_status"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    chosen: Mapped[str] = mapped_column(Text, nullable=False)
    rejected: Mapped[str] = mapped_column(Text, nullable=False)
    rubric: Mapped[Optional[str]] = mapped_column(Text)
    annotator: Mapped[Optional[str]] = mapped_column(Text)
    metadata_: Mapped[Optional[str]] = mapped_column("metadata", Text)
    review_status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 25. 适配器兼容性表
# ============================================

class AdapterCompatibility(Base):
    """LoRA 适配器兼容性检查记录表"""
    __tablename__ = "adapter_compatibility"
    __table_args__ = (
        Index("idx_adapter_compat_name", "adapter_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adapter_name: Mapped[str] = mapped_column(Text, nullable=False)
    checked_at: Mapped[str] = mapped_column(Text, nullable=False)
    compatible: Mapped[int] = mapped_column(Integer, nullable=False)
    checks: Mapped[Optional[str]] = mapped_column(Text)
    warnings: Mapped[Optional[str]] = mapped_column(Text)
    errors: Mapped[Optional[str]] = mapped_column(Text)


# ============================================
# 26. 用户反馈表
# ============================================

class Feedback(Base):
    """用户反馈表（thumbs up/down 等）"""
    __tablename__ = "feedback"
    __table_args__ = (
        Index("idx_feedback_created", "created_at"),
        Index("idx_feedback_trace_id", "trace_id"),
        Index("idx_feedback_message_id", "message_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[Optional[str]] = mapped_column(Text)
    message_id: Mapped[Optional[str]] = mapped_column(Text)
    rating: Mapped[Optional[str]] = mapped_column(Text)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    adapter_name: Mapped[Optional[str]] = mapped_column(Text)
    kb_revision: Mapped[Optional[str]] = mapped_column(Text)
    prompt_version: Mapped[Optional[str]] = mapped_column(Text)
    detail: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 27. 角色关系表
# ============================================

class CharacterRelationship(Base):
    """角色关系表：人物与当前聊天用户的关系状态。

    主键为完整的记忆隔离范围（平台+适配器+发送者+会话类型+会话ID），
    与 character.models.UserScope.memory_scope_key 语义一致：
    私聊按用户隔离，群聊按"群+用户"隔离。
    """
    __tablename__ = "character_relationships"

    character_id: Mapped[str] = mapped_column(Text, primary_key=True)
    platform: Mapped[str] = mapped_column(Text, primary_key=True)
    adapter: Mapped[str] = mapped_column(Text, primary_key=True)
    sender_id: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_type: Mapped[str] = mapped_column(Text, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    relationship_stage: Mapped[str] = mapped_column(Text, nullable=False)
    preferred_address: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    interaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


# ============================================
# 28. 角色长期记忆表
# ============================================

class CharacterMemory(Base):
    """角色长期记忆表：按角色+用户范围隔离的用户事实与共同经历。

    memory_key 在同一范围内唯一（upsert 键），例如
    "user_name"、"preference_甜食"，防止同一事实反复入库。
    """
    __tablename__ = "character_memories"
    __table_args__ = (
        Index(
            "idx_character_memories_scope",
            "character_id",
            "platform",
            "adapter",
            "sender_id",
            "conversation_type",
            "conversation_id",
        ),
        UniqueConstraint(
            "character_id",
            "platform",
            "adapter",
            "sender_id",
            "conversation_type",
            "conversation_id",
            "memory_key",
            name="uq_character_memory_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    character_id: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    adapter: Mapped[str] = mapped_column(Text, nullable=False)
    sender_id: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_type: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_id: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(Text, nullable=False)
    memory_key: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    source_message_id: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
