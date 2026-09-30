"""Index source-to-claim conflict checks for targeted source erasure."""
from alembic import op

revision = '014_source_link_lookup'
down_revision = '013_memory_source_terms'
branch_labels = None
depends_on = None


def upgrade():
    op.execute('CREATE INDEX IF NOT EXISTS idx_memory_source_links_source ON memory_source_links(source_key)')


def downgrade():
    op.execute('DROP INDEX IF EXISTS idx_memory_source_links_source')
