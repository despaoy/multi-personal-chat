"""Persist explicit character identity independently of the model adapter."""
import sqlalchemy as sa

from alembic import op

revision = '010_message_character_scope'
down_revision = '009_narrative_branches'
branch_labels = None
depends_on = None


def upgrade():
    columns = {c['name'] for c in sa.inspect(op.get_bind()).get_columns('messages')}
    if 'characterId' not in columns:
        op.add_column('messages', sa.Column('characterId', sa.Text(), nullable=True))
    # Do not guess identity from a mutable LoRA mapping or assign legacy rows
    # to whichever character happens to be selected during migration.


def downgrade():
    op.drop_column('messages', 'characterId')
