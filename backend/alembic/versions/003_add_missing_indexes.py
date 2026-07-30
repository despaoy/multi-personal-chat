"""补充缺失索引（与 SQLite 运行时初始化对齐）

Revision ID: 003_indexes
Revises: 002_research
Create Date: 2026-07-30 00:00:00.000000

此前仅在 SQLite 的 _init_database() 运行时 CREATE INDEX IF NOT EXISTS
中创建这些索引，PostgreSQL 部署通过 ORM metadata.create_all() 或
Alembic 迁移得不到这些优化，违背"ORM 是 schema 唯一权威"的目标。
本迁移将索引补到 PostgreSQL，使两端 schema 一致。

幂等性：服务器先启动新代码（PgDatabase.init() 会创建索引）再执行
alembic upgrade 时，无条件 op.create_index() 会因索引已存在而失败。
因此使用 SQLAlchemy Inspector 先检查索引是否存在，存在则跳过。
SQLite 的 CREATE INDEX IF NOT EXISTS 天然幂等，但 Alembic 的
op.create_index 不带 IF NOT EXISTS，故两端统一走 Inspector 检查。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '003_indexes'
down_revision: Union[str, None] = '002_research'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_exists(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    """检查索引是否已存在（幂等迁移）。"""
    try:
        indexes = inspector.get_indexes(table_name)
    except Exception:
        return False
    return any(idx["name"] == index_name for idx in indexes)


def _create_index_if_missing(
    inspector: sa.Inspector,
    index_name: str,
    table_name: str,
    columns: list,
) -> None:
    """如果索引不存在则创建，存在则跳过。"""
    if _index_exists(inspector, table_name, index_name):
        return
    op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # 知识库文档外键索引
    _create_index_if_missing(inspector, 'idx_knowledge_documents_kb_id', 'knowledge_documents', ['knowledge_base_id'])
    _create_index_if_missing(inspector, 'idx_knowledge_documents_folder_id', 'knowledge_documents', ['folder_id'])
    # 知识库分块外键索引
    _create_index_if_missing(inspector, 'idx_knowledge_chunks_documentId', 'knowledge_chunks', ['documentId'])
    # 训练任务过滤索引
    _create_index_if_missing(inspector, 'idx_training_tasks_lora_name', 'training_tasks', ['lora_name'])
    _create_index_if_missing(inspector, 'idx_training_tasks_status', 'training_tasks', ['status'])
    # 意图样本按知识库过滤
    _create_index_if_missing(inspector, 'idx_intent_samples_kbName', 'intent_samples', ['kbName'])
    # 审计日志按时间排序
    _create_index_if_missing(inspector, 'idx_audit_logs_timestamp', 'audit_logs', ['timestamp'])
    # 反馈按 trace_id/message_id 查询
    _create_index_if_missing(inspector, 'idx_feedback_trace_id', 'feedback', ['trace_id'])
    _create_index_if_missing(inspector, 'idx_feedback_message_id', 'feedback', ['message_id'])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    def _drop_if_exists(index_name: str, table_name: str) -> None:
        if _index_exists(inspector, table_name, index_name):
            op.drop_index(index_name, table_name=table_name)

    _drop_if_exists('idx_feedback_message_id', 'feedback')
    _drop_if_exists('idx_feedback_trace_id', 'feedback')
    _drop_if_exists('idx_audit_logs_timestamp', 'audit_logs')
    _drop_if_exists('idx_intent_samples_kbName', 'intent_samples')
    _drop_if_exists('idx_training_tasks_status', 'training_tasks')
    _drop_if_exists('idx_training_tasks_lora_name', 'training_tasks')
    _drop_if_exists('idx_knowledge_chunks_documentId', 'knowledge_chunks')
    _drop_if_exists('idx_knowledge_documents_folder_id', 'knowledge_documents')
    _drop_if_exists('idx_knowledge_documents_kb_id', 'knowledge_documents')
