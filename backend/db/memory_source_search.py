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


def literal_project_source_terms(query):
    """Return unambiguous literal project tags for an explicit source read.

    This is a lexical task scope, not a claim about semantic relevance or
    completeness. Mixed tasks, excluded named projects and detached references
    defer unchanged to the existing unrestricted sparse query.
    """
    if (not isinstance(query, str)
            or not re.search(r"列出|读取|查看|查找|复述|还原", query)
            or not re.search(r"原话|原始.{0,6}(?:记录|资料)|交接记录", query)
            or re.search(r"以及|同时|另外|还要|顺便|并且", query)):
        return ()
    code = r"[A-Za-z]+[0-9][A-Za-z0-9]*"
    matches = list(re.finditer(rf"(?<![A-Za-z0-9_-])({code})\s*项目", query))
    if not matches:
        return ()
    if any(re.search(r"不含|不包括|排除|不要|别查|除去|不是", query[max(0, m.start()-8):m.start()]) for m in matches):
        return ()
    tags = tuple(sorted({m.group(1).lower() for m in matches}))
    identifiers = re.findall(rf"(?<![A-Za-z0-9_-])({code}(?:[-_][A-Za-z0-9]+)*)(?![A-Za-z0-9_-])", query)
    if any(not any(identity.lower() == tag or identity.lower().startswith((tag+'-', tag+'_')) for tag in tags)
           for identity in identifiers):
        return ()
    return tags


def literal_source_fragments(query):
    """Union of independently closed positive literal source reads.

    Each leading unquoted declaration names its own complete raw record scope.
    Do not guess a conjunctive locator, an unresolved extra read or a quoted
    command. Exact fragments identify speech, not a subject or verified fact.
    """
    from character.quoted_erasure_authority import masked_quotes

    if not isinstance(query, str):
        return ()
    try:
        view, quotes = masked_quotes(query)
    except ValueError:
        return ()
    fragments, position = [], 0
    prefix = (r"\s*(?:(?:再|另|同时|另外)?(?:请)?|请(?:再|另|同时|另外)?)"
              r"(?:查找|检索|读取|查看)\s*(?:包含|含有)\s*")
    for a, z, fragment in quotes:
        if not re.fullmatch(prefix, view[position:a]):
            break
        if (not fragment.strip() or "\n" in fragment or not terms(fragment)
                or any(char in fragment for char in '“”‘’「」『』"')):
            return ()
        read = re.match(r"\s*的\s*(?:完整|全部)?(?:用户)?(?:原话|原始发言)(?:记录)?\s*(?:[。；;\n]|$)", view[z:])
        if not read:
            return ()
        fragments.append(fragment)
        position = z + read.end()
    remainder = view[position:]
    if (not fragments or re.search(r"另外|同时|以及|还要|并且|顺便", remainder)
            or re.search(r"(?:查找|检索|读取|查看|复述|还原)[^。；;\n]*(?:原话|原始|记录)", remainder)):
        return ()
    return tuple(dict.fromkeys(fragments))


def literal_source_fragment(query):
    """Keep the existing single-locator API: plural reads are not one locator."""
    fragments = literal_source_fragments(query)
    return fragments[0] if len(fragments) == 1 else ""



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
    fragments = literal_source_fragments(query)
    project_tags = () if fragments else literal_project_source_terms(query)
    project_filter = ""
    if project_tags:
        params['project_terms'] = json.dumps(project_tags)
        project_bound = ('SELECT value FROM json_each(:project_terms)' if dialect == 'sqlite' else
                         'SELECT value FROM jsonb_array_elements_text(CAST(:project_terms AS JSONB)) AS p(value)')
        project_filter = ("JOIN (SELECT DISTINCT anchor.source_key FROM memory_source_terms anchor "
                          "WHERE anchor.scope_key = :scope_key "
                          f"AND anchor.term IN ({project_bound})) project_scope "
                          "ON project_scope.source_key = t.source_key ")
    fragment_filter = ""
    if fragments:
        expressions = []
        for index, fragment in enumerate(fragments):
            name = f"source_fragment{index}"
            params[name] = fragment
            expressions.append(f"instr(eligible.body, :{name}) > 0" if dialect == "sqlite" else
                               f"POSITION(:{name} IN eligible.body) > 0")
        fragment_filter = "AND (" + " OR ".join(expressions) + ") "
    # Inverse posting frequency is a retrieval weight, not semantic confidence.
    # Full source rows are reached through scoped postings, not latest-N scans.
    # Erasure watermark is checked in this same SQL snapshot; no text cache.
    rows = yield ("WITH hits AS (SELECT t.source_key, t.term FROM memory_source_terms t "
                  f"{project_filter}"
                  "JOIN memory_sources eligible ON eligible.source_key = t.source_key "
                  f"WHERE t.scope_key = :scope_key AND t.term IN ({bound}) "
                  f"{fragment_filter}"
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
