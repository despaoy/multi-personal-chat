"""初始数据库模式

与 db/pg_database.py 中的 SQLAlchemy Core 表定义保持一致。
此前版本列名/数量/类型与实际 schema 严重不符，导致 alembic upgrade head
创建的表无法被应用使用。本迁移重写为与运行时 schema 完全一致。

Revision ID: 001_initial
Revises: None
Create Date: 2025-01-01 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ============================================
    # 用户表
    # ============================================
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('username', sa.Text(), nullable=False, unique=True),
        sa.Column('password_hash', sa.Text(), nullable=False),
        sa.Column('created_at', sa.Text(), nullable=False),
        # role 列：RBAC 权限系统依赖。默认 'user'，首个用户由应用层升级为 'admin'。
        sa.Column('role', sa.Text(), nullable=False, server_default='user'),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 配置表
    # ============================================
    op.create_table(
        'config',
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('key'),
    )

    # ============================================
    # 消息表（列名与 pg_database.py messages_table 完全一致）
    # ============================================
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('sessionType', sa.Text(), nullable=False),
        sa.Column('sessionId', sa.Text(), nullable=False),
        sa.Column('sessionName', sa.Text(), nullable=True),
        sa.Column('platform', sa.Text(), nullable=False, server_default='qq'),
        sa.Column('adapter', sa.Text(), nullable=False, server_default='nonebot'),
        sa.Column('conversationId', sa.Text(), nullable=True),
        sa.Column('conversationType', sa.Text(), nullable=True),
        sa.Column('senderId', sa.Text(), nullable=True),
        sa.Column('senderName', sa.Text(), nullable=True),
        sa.Column('sourceMessageId', sa.Text(), nullable=True),
        sa.Column('traceId', sa.Text(), nullable=True),
        sa.Column('userId', sa.Text(), nullable=True),
        sa.Column('userName', sa.Text(), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('reply', sa.Text(), nullable=False),
        sa.Column('modelName', sa.Text(), nullable=True),
        sa.Column('loraName', sa.Text(), nullable=True),
        sa.Column('costTime', sa.Float(), nullable=True),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # LoRA 模型表（id 为 Text，与 pg_database.py 一致）
    # ============================================
    op.create_table(
        'loras',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.Text(), nullable=False, server_default='inactive'),
        sa.Column('style', sa.Text(), nullable=True),
        sa.Column('size', sa.Text(), nullable=True),
        sa.Column('trainedSteps', sa.Integer(), nullable=True),
        sa.Column('totalSteps', sa.Integer(), nullable=True),
        sa.Column('createdAt', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 知识库表
    # ============================================
    op.create_table(
        'knowledge_bases',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.Text(), nullable=False, unique=True),
        sa.Column('description', sa.Text(), server_default=''),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 知识库文件夹表
    # ============================================
    op.create_table(
        'knowledge_folders',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('knowledge_base_id', sa.Integer(),
                  sa.ForeignKey('knowledge_bases.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), server_default=''),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('knowledge_base_id', 'name'),
    )

    # ============================================
    # 知识库文档表
    # ============================================
    op.create_table(
        'knowledge_documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('category', sa.Text(), nullable=False, server_default='未分类'),
        sa.Column('knowledge_base_id', sa.Integer(),
                  sa.ForeignKey('knowledge_bases.id', ondelete='SET NULL'), nullable=True),
        sa.Column('folder_id', sa.Integer(),
                  sa.ForeignKey('knowledge_folders.id', ondelete='SET NULL'), nullable=True),
        sa.Column('sourceType', sa.Text(), nullable=False, server_default='text'),
        sa.Column('sourceUrl', sa.Text(), nullable=True),
        sa.Column('fileType', sa.Text(), nullable=True),
        sa.Column('fileSize', sa.Integer(), nullable=True),
        sa.Column('chunkCount', sa.Integer(), server_default='0'),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.Column('updatedAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 知识库文档分块表
    # ============================================
    op.create_table(
        'knowledge_chunks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('documentId', sa.Integer(),
                  sa.ForeignKey('knowledge_documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunkIndex', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        # embedding 存储 Faiss 向量 ID（BigInteger），不是 BLOB
        sa.Column('embedding', sa.BigInteger(), nullable=True),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 用户数据持久化表
    # ============================================
    op.create_table(
        'user_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('page_key', sa.Text(), nullable=False),
        sa.Column('data_json', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'page_key'),
    )

    # ============================================
    # Claw 工具表
    # ============================================
    op.create_table(
        'claw_tools',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.Text(), nullable=False, unique=True),
        sa.Column('description', sa.Text(), server_default=''),
        sa.Column('code', sa.Text(), server_default=''),
        sa.Column('enabled', sa.Integer(), server_default='1'),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 集成消息去重表
    # ============================================
    op.create_table(
        'integration_message_dedup',
        sa.Column('dedupKey', sa.Text(), nullable=False),
        sa.Column('platform', sa.Text(), nullable=False),
        sa.Column('adapter', sa.Text(), nullable=False),
        sa.Column('messageId', sa.Text(), nullable=False),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('dedupKey'),
    )

    # ============================================
    # 会话聚合表
    # ============================================
    op.create_table(
        'conversations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.Text(), nullable=False),
        sa.Column('conversationId', sa.Text(), nullable=False),
        sa.Column('conversationType', sa.Text(), nullable=False, server_default='private'),
        sa.Column('displayName', sa.Text(), nullable=True),
        sa.Column('botEnabled', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('replyPolicy', sa.Text(), nullable=False, server_default='default'),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.Column('updatedAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('platform', 'conversationId', 'conversationType',
                            name='uq_conversations_platform_conversation'),
    )

    # ============================================
    # 集成事件表
    # ============================================
    op.create_table(
        'integration_events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('platform', sa.Text(), nullable=False),
        sa.Column('adapter', sa.Text(), nullable=False),
        sa.Column('sourceMessageId', sa.Text(), nullable=True),
        sa.Column('conversationId', sa.Text(), nullable=True),
        sa.Column('conversationType', sa.Text(), nullable=True),
        sa.Column('senderId', sa.Text(), nullable=True),
        sa.Column('eventType', sa.Text(), nullable=False, server_default='message'),
        sa.Column('eventHash', sa.Text(), nullable=False),
        sa.Column('rawSummary', sa.Text(), nullable=True),
        sa.Column('traceId', sa.Text(), nullable=True),
        sa.Column('status', sa.Text(), nullable=False, server_default='received'),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('platform', 'adapter', 'eventHash',
                            name='uq_integration_events_platform_hash'),
    )

    # ============================================
    # 模型调用记录表
    # ============================================
    op.create_table(
        'model_invocations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('traceId', sa.Text(), nullable=True),
        sa.Column('platform', sa.Text(), server_default='qq'),
        sa.Column('conversationId', sa.Text(), nullable=True),
        sa.Column('sessionId', sa.Text(), nullable=True),
        sa.Column('modelName', sa.Text(), nullable=True),
        sa.Column('loraName', sa.Text(), nullable=True),
        sa.Column('costTime', sa.Float(), server_default='0'),
        sa.Column('promptTokens', sa.Integer(), server_default='0'),
        sa.Column('completionTokens', sa.Integer(), server_default='0'),
        sa.Column('totalTokens', sa.Integer(), server_default='0'),
        sa.Column('usedRag', sa.Integer(), server_default='0'),
        sa.Column('usedLora', sa.Integer(), server_default='0'),
        sa.Column('errorType', sa.Text(), server_default=''),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 审计日志表
    # ============================================
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('api_key_hash', sa.Text(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('action', sa.Text(), nullable=False),
        sa.Column('resource', sa.Text(), nullable=True),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 意图样本表
    # ============================================
    op.create_table(
        'intent_samples',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('kbName', sa.Text(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('label', sa.Text(), nullable=False),
        sa.Column('createdAt', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 意图活跃知识库表
    # ============================================
    op.create_table(
        'intent_active_kbs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('kbName', sa.Text(), nullable=False),
        sa.Column('isActive', sa.Integer(), server_default='1'),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 训练任务表
    # 包含新列（task_id, lora_name, status 等）和 legacy 列（taskType, config 等）
    # ============================================
    op.create_table(
        'training_tasks',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('task_id', sa.Text(), nullable=True),
        sa.Column('lora_name', sa.Text(), server_default=''),
        sa.Column('status', sa.Text(), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Float(), server_default='0'),
        sa.Column('error_message', sa.Text(), server_default=''),
        sa.Column('config_json', sa.Text(), server_default='{}'),
        sa.Column('created_at', sa.Text(), server_default=''),
        sa.Column('updated_at', sa.Text(), server_default=''),
        # Legacy columns（保留兼容已有 PG 部署）
        sa.Column('taskType', sa.Text(), server_default='lora'),
        sa.Column('config', sa.Text(), nullable=True),
        sa.Column('result', sa.Text(), nullable=True),
        sa.Column('createdAt', sa.Text(), server_default=''),
        sa.Column('updatedAt', sa.Text(), server_default=''),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 保存的对话表
    # ============================================
    op.create_table(
        'saved_dialogues',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('character_desc', sa.Text(), nullable=False),
        sa.Column('style', sa.Text(), nullable=True),
        sa.Column('dialogue_count', sa.Integer(), server_default='0'),
        sa.Column('dialogues_json', sa.Text(), nullable=False),
        sa.Column('turn_stats', sa.Text(), nullable=True),
        sa.Column('scene_stats', sa.Text(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    # ============================================
    # 索引（与 pg_database.py init() 中的索引一致）
    # ============================================
    op.create_index(
        'idx_training_tasks_task_id', 'training_tasks', ['task_id'], unique=True,
    )
    op.create_index(
        'idx_messages_platform_conversation', 'messages',
        ['platform', 'conversationId', 'createdAt'],
    )
    op.create_index(
        'idx_messages_source_dedup', 'messages',
        ['platform', 'adapter', 'sourceMessageId'],
    )
    op.create_index(
        'idx_messages_session_created', 'messages', ['sessionId', 'createdAt'],
    )
    op.create_index(
        'idx_messages_created_at', 'messages', ['createdAt'],
    )
    op.create_index(
        'idx_conversations_platform_conversation', 'conversations',
        ['platform', 'conversationId', 'conversationType'],
    )
    op.create_index(
        'idx_integration_events_trace', 'integration_events', ['traceId'],
    )
    op.create_index(
        'idx_integration_events_platform_created', 'integration_events',
        ['platform', 'createdAt'],
    )
    op.create_index(
        'idx_model_invocations_trace', 'model_invocations', ['traceId'],
    )
    op.create_index(
        'idx_model_invocations_created', 'model_invocations', ['createdAt'],
    )


def downgrade() -> None:
    # 删除索引
    op.drop_index('idx_model_invocations_created', table_name='model_invocations')
    op.drop_index('idx_model_invocations_trace', table_name='model_invocations')
    op.drop_index('idx_integration_events_platform_created', table_name='integration_events')
    op.drop_index('idx_integration_events_trace', table_name='integration_events')
    op.drop_index('idx_conversations_platform_conversation', table_name='conversations')
    op.drop_index('idx_messages_created_at', table_name='messages')
    op.drop_index('idx_messages_session_created', table_name='messages')
    op.drop_index('idx_messages_source_dedup', table_name='messages')
    op.drop_index('idx_messages_platform_conversation', table_name='messages')
    op.drop_index('idx_training_tasks_task_id', table_name='training_tasks')

    # 按依赖逆序删除表
    op.drop_table('saved_dialogues')
    op.drop_table('training_tasks')
    op.drop_table('intent_active_kbs')
    op.drop_table('intent_samples')
    op.drop_table('audit_logs')
    op.drop_table('model_invocations')
    op.drop_table('integration_events')
    op.drop_table('conversations')
    op.drop_table('integration_message_dedup')
    op.drop_table('claw_tools')
    op.drop_table('user_data')
    op.drop_table('knowledge_chunks')
    op.drop_table('knowledge_documents')
    op.drop_table('knowledge_folders')
    op.drop_table('knowledge_bases')
    op.drop_table('loras')
    op.drop_table('messages')
    op.drop_table('config')
    op.drop_table('users')
