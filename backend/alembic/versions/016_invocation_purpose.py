"""Separate chat, knowledge and background accounting without reclassifying history."""
from alembic import op
import sqlalchemy as sa

revision = '016_invocation_purpose'
down_revision = '015_invocation_usage_source'
branch_labels = None
depends_on = None


def upgrade():
    columns = {column['name'] for column in sa.inspect(op.get_bind()).get_columns('model_invocations')}
    if 'purpose' not in columns:
        op.add_column('model_invocations', sa.Column('purpose', sa.Text(), nullable=False, server_default='chat'))


def downgrade():
    op.drop_column('model_invocations', 'purpose')
