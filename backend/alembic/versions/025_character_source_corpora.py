"""Immutable account-owned original source editions."""

import sqlalchemy as sa

from alembic import op

revision = '025_character_source_corpora'
down_revision = '024_chat_memory_vectors'
branch_labels = None
depends_on = None


def upgrade():
    if 'character_source_corpora' in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        'character_source_corpora',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('owner_user_id', sa.Text(), nullable=False),
        sa.Column('character_id', sa.Text(), sa.ForeignKey('character_library.id', ondelete='CASCADE'), nullable=False),
        sa.Column('character_revision', sa.Integer(), nullable=False),
        sa.Column('content_digest', sa.Text(), nullable=False),
        sa.Column('sources_json', sa.Text(), nullable=False),
        sa.Column('revoked', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.Text(), nullable=False),
    )
    op.create_index('ix_character_source_scope', 'character_source_corpora', ['owner_user_id', 'character_id', 'character_revision'])


def downgrade():
    op.drop_table('character_source_corpora')
