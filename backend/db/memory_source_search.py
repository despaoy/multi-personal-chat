"""Portable scope-first inverted index over source speech, not display labels."""
from __future__ import annotations

import json
import re

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS memory_source_terms (
        scope_key TEXT NOT NULL, term TEXT NOT NULL, source_key TEXT NOT NULL,
        PRIMARY KEY (scope_key, term, source_key))""",
    "CREATE INDEX IF NOT EXISTS idx_memory_source_terms_source ON memory_source_terms(source_key)",
)


def terms(text):
    result = set()
    for token in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            result.update(token[i:i + 2] for i in range(len(token) - 1))
        elif len(token) > 1:
            result.add(token)
    return tuple(sorted(result))


def index_plan(identity, body):
    tokens = terms(body)
    for offset in range(0, len(tokens), 100):
        params = dict(identity)
        values = []
        for index, token in enumerate(tokens[offset:offset + 100]):
            params[f"term{index}"] = token
            values.append(f"(:scope_key, :term{index}, :source_key)")
        yield ("INSERT INTO memory_source_terms (scope_key, term, source_key) VALUES "
               + ",".join(values) + " ON CONFLICT DO NOTHING", params)


def search_plan(scope, query, *, limit=32, dialect='sqlite'):
    # None is the internal complete-search contract for serving-budget callers.
    # Explicit bounded callers keep their existing validation and SQL limit.
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 100):
        raise ValueError("Source search limit must be 1..100")
    if not isinstance(query, str):
        raise ValueError("Source query must be text")
    # Loaded user history can exceed the request-message limit. All terms use
    # one JSON binding below; do not truncate a complete available topic.
    if dialect not in {'sqlite', 'postgres'}:
        raise ValueError('Unsupported source search dialect')
    tokens = terms(query)
    if not tokens:
        return []
    params = dict(scope, limit=limit)
    # One bound JSON array avoids variable-count limits without choosing a
    # prefix/subset of terms. Keep global ranking in one database snapshot.
    params['query_terms'] = json.dumps(tokens, ensure_ascii=False)
    bound = ('SELECT value FROM json_each(:query_terms)' if dialect == 'sqlite' else
             'SELECT value FROM jsonb_array_elements_text(CAST(:query_terms AS JSONB)) AS q(value)')
    # Inverse posting frequency is a retrieval weight, not semantic confidence.
    # Full source rows are reached through scoped postings, not latest-N scans.
    # Erasure watermark is checked in this same SQL snapshot; no text cache.
    rows = yield ("WITH hits AS (SELECT t.source_key, t.term FROM memory_source_terms t "
                  "JOIN memory_sources eligible ON eligible.source_key = t.source_key "
                  f"WHERE t.scope_key = :scope_key AND t.term IN ({bound}) "
                  "AND eligible.state = 'recorded' AND eligible.observed_at > "
                  "COALESCE((SELECT revoked_before FROM memory_source_fences WHERE owner_key = :owner_key), '')), "
                  "df AS (SELECT term, COUNT(*) AS n FROM hits GROUP BY term), "
                  "ranked AS (SELECT h.source_key, SUM(CAST(1 AS DOUBLE PRECISION) / df.n) AS score, COUNT(*) AS matched_terms "
                  "FROM hits h JOIN df ON h.term = df.term GROUP BY h.source_key) "
                  "SELECT s.source_message_id, s.observed_at, s.body, r.score, r.matched_terms "
                  "FROM ranked r JOIN memory_sources s ON s.source_key = r.source_key "
                  "WHERE s.scope_key = :scope_key AND s.state = 'recorded' AND s.observed_at > "
                  "COALESCE((SELECT revoked_before FROM memory_source_fences WHERE owner_key = :owner_key), '') "
                  "ORDER BY r.score DESC, s.observed_at DESC, s.source_key DESC"
                  + (" LIMIT :limit" if limit is not None else ""), params)
    return rows
