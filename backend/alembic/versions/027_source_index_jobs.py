"""Durable source indexing attempts, including pre-existing completed receipts."""

import sqlalchemy as sa

from alembic import op

revision = '027_source_index_jobs'
down_revision = '026_source_index_receipt'
branch_labels = None
depends_on = None


def upgrade():
    columns = {column['name'] for column in sa.inspect(op.get_bind()).get_columns('character_source_corpora')}
    if 'index_status' not in columns:
        op.add_column('character_source_corpora', sa.Column('index_status', sa.Text(), nullable=False, server_default='pending'))
    if 'index_attempt' not in columns:
        op.add_column('character_source_corpora', sa.Column('index_attempt', sa.Integer(), nullable=False, server_default='1'))
    op.execute("UPDATE character_source_corpora SET index_status='completed' WHERE index_receipt_json IS NOT NULL AND index_status='pending'")


def downgrade():
    op.drop_column('character_source_corpora', 'index_attempt')
    op.drop_column('character_source_corpora', 'index_status')
