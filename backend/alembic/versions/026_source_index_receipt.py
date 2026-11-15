"""Persist only completely acknowledged source index publications."""

import sqlalchemy as sa

from alembic import op

revision = '026_source_index_receipt'
down_revision = '025_character_source_corpora'
branch_labels = None
depends_on = None


def upgrade():
    if 'index_receipt_json' not in {column['name'] for column in sa.inspect(op.get_bind()).get_columns('character_source_corpora')}:
        op.add_column('character_source_corpora', sa.Column('index_receipt_json', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('character_source_corpora', 'index_receipt_json')
