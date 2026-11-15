"""Account-owned characters and immutable settings snapshots."""

import sqlalchemy as sa

from alembic import op

revision = "017_character_library"
down_revision = "016_invocation_purpose"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    metadata = sa.MetaData()
    characters = sa.Table(
        "character_library",
        metadata,
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("owner_user_id", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_character_id", sa.Text()),
        sa.Column("source_revision", sa.Integer()),
        sa.Column("share_token", sa.Text(), unique=True),
        sa.Column("shared_revision", sa.Integer()),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.CheckConstraint("revision >= 1", name="ck_character_library_revision"),
    )
    sa.Index("ix_character_library_owner_user_id", characters.c.owner_user_id)
    sa.Table(
        "character_library_versions",
        metadata,
        sa.Column(
            "character_id", sa.Text(), sa.ForeignKey("character_library.id", ondelete="CASCADE"), primary_key=True
        ),
        sa.Column("revision", sa.Integer(), primary_key=True),
        sa.Column("settings_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    metadata.create_all(bind, checkfirst=True)


def downgrade():
    op.drop_table("character_library_versions")
    op.drop_table("character_library")
