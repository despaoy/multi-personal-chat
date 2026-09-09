"""Add durable gateway generation/delivery receipts."""

from alembic import op

revision = "008_integration_receipts"
down_revision = "007_memory_claims"
branch_labels = None
depends_on = None


def upgrade():
    from db.integration_receipts import CREATE_SQL

    op.execute(CREATE_SQL)
    op.execute("CREATE INDEX IF NOT EXISTS idx_integration_receipt_owner ON integration_receipts(owner)")


def downgrade():
    op.drop_table("integration_receipts")
