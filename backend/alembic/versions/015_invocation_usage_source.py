"""Identify measured invocation usage without rewriting historical estimates."""
from alembic import op
import sqlalchemy as sa

revision = '015_invocation_usage_source'
down_revision = '014_source_link_lookup'
branch_labels = None
depends_on = None


def upgrade():
    columns = {column['name'] for column in sa.inspect(op.get_bind()).get_columns('model_invocations')}
    if 'usageSource' not in columns:
        op.add_column('model_invocations', sa.Column('usageSource', sa.Text(), nullable=False, server_default='legacy_unverified'))


def downgrade():
    op.drop_column('model_invocations', 'usageSource')
