"""Immutable selected-source publications and idempotent recipient copies."""
import sqlalchemy as sa

from alembic import op

revision = '028_character_publications'
down_revision = '027_source_index_jobs'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {c['name'] for c in sa.inspect(bind).get_columns('character_library')}
    for name in ('publication_id', 'source_publication_id'):
        if name not in columns:
            op.add_column('character_library', sa.Column(name, sa.Text()))
    metadata = sa.MetaData()
    sa.Table('character_publications', metadata,
             sa.Column('id', sa.Text(), primary_key=True),
             sa.Column('character_id', sa.Text(), nullable=False),
             sa.Column('character_revision', sa.Integer(), nullable=False),
             sa.Column('corpus_id', sa.Text()),
             sa.Column('settings_json', sa.Text(), nullable=False),
             sa.Column('sources_json', sa.Text(), nullable=False),
             sa.Column('created_at', sa.Text(), nullable=False))
    sa.Table('character_publication_copies', metadata,
             sa.Column('owner_user_id', sa.Text(), primary_key=True),
             sa.Column('request_id', sa.Text(), primary_key=True),
             sa.Column('publication_id', sa.Text(), nullable=False),
             sa.Column('character_id', sa.Text(), nullable=False))
    metadata.create_all(bind, checkfirst=True)
    # Existing settings-only shares remain accessible, never opt in existing corpora.
    op.execute("INSERT INTO character_publications (id,character_id,character_revision,settings_json,sources_json,created_at) "
               "SELECT 'legacy-' || c.id || '-' || CAST(c.shared_revision AS TEXT),c.id,c.shared_revision,v.settings_json,'[]',c.updated_at "
               "FROM character_library c JOIN character_library_versions v ON v.character_id=c.id AND v.revision=c.shared_revision "
               "WHERE c.share_token IS NOT NULL AND c.publication_id IS NULL "
               "AND NOT EXISTS (SELECT 1 FROM character_publications p WHERE p.id='legacy-' || c.id || '-' || CAST(c.shared_revision AS TEXT))")
    op.execute("UPDATE character_library SET publication_id='legacy-' || id || '-' || CAST(shared_revision AS TEXT) "
               "WHERE share_token IS NOT NULL AND publication_id IS NULL")


def downgrade():
    op.drop_table('character_publication_copies')
    op.drop_table('character_publications')
    op.drop_column('character_library', 'source_publication_id')
    op.drop_column('character_library', 'publication_id')
