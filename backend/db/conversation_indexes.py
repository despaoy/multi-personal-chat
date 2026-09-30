"""Fixed indexes for scope-first chronological source reads.

No automatic DDL on reads. Bootstrap/migration callers own installation.
"""

SOURCE_INDEX_STATEMENTS = (
    '''CREATE INDEX IF NOT EXISTS idx_messages_private_source
       ON messages (platform, adapter, "senderId", "characterId", "createdAt" DESC, id DESC)
       WHERE "branchId" IS NULL AND "conversationType" IN ('private', '')''',
    '''CREATE INDEX IF NOT EXISTS idx_messages_local_source
       ON messages (platform, adapter, "senderId", "characterId", "conversationType",
                    "conversationId", "createdAt" DESC, id DESC)
       WHERE "branchId" IS NULL''',
)
