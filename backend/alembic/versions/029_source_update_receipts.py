"""Bind successful source updates to exact final previews and choices."""
import sqlalchemy as sa

from alembic import op

revision = '029_source_update_receipts'
down_revision = '028_character_publications'
branch_labels = None
depends_on = None


def upgrade():
    metadata = sa.MetaData()
    sa.Table('character_source_update_receipts', metadata,
             sa.Column('owner_user_id', sa.Text(), primary_key=True),
             sa.Column('character_id', sa.Text(), primary_key=True),
             sa.Column('preview_id', sa.Text(), primary_key=True),
             sa.Column('request_json', sa.Text(), nullable=False),
             sa.Column('result_revision', sa.Integer(), nullable=False))
    metadata.create_all(op.get_bind(), checkfirst=True)


def downgrade():
    op.drop_table('character_source_update_receipts')
