"""Portable SQL for durable gateway generation and delivery receipts."""

CREATE_SQL = """CREATE TABLE IF NOT EXISTS integration_receipts (
    receipt_key TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,
    response TEXT NOT NULL DEFAULT '',
    expires_at DOUBLE PRECISION NOT NULL
)"""

STATEMENTS = {
    "archive": """INSERT INTO integration_receipts (receipt_key, owner, status, response, expires_at)
        SELECT 'attempt:' || owner, owner, 'superseded', '', expires_at FROM integration_receipts
        WHERE receipt_key = :key AND owner <> :owner
          AND (status = 'failed' OR (status = 'processing' AND expires_at < :now))
        ON CONFLICT (receipt_key) DO NOTHING""",
    "get": "SELECT * FROM integration_receipts WHERE receipt_key = :key",
    "claim": """INSERT INTO integration_receipts (receipt_key, owner, status, response, expires_at)
        VALUES (:key, :owner, 'processing', '', :expires_at)
        ON CONFLICT (receipt_key) DO UPDATE SET owner = :owner, status = 'processing',
            response = '', expires_at = :expires_at
        WHERE integration_receipts.status = 'failed'
           OR (integration_receipts.status = 'processing' AND integration_receipts.expires_at < :now)""",
    "finish": """UPDATE integration_receipts SET status = :status, response = :response
        WHERE receipt_key = :key AND owner = :owner AND status = 'processing' """,
    "delivery": """UPDATE integration_receipts SET status = :status
        WHERE receipt_key = :key AND owner = :owner AND status IN ('generated', 'delivery_failed')""",
}

# Pending or failed sends must not appear as conversation the user has seen.
HISTORY_DELIVERY_FILTER = """NOT EXISTS (
    SELECT 1 FROM integration_receipts r
    WHERE r.owner = messages."traceId" AND r.status <> 'delivered'
)"""
