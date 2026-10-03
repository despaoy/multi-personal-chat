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



def unresolved_source_read_tail(view):
    """Distinguish a source-read action from a reference to its purpose.

    The caller already masked quoted material and closed every leading read.
    Only a bare or explicitly referring source noun followed by a restrictive
    purpose marker is a nominal explanation. Extra/negative reads and unknown
    additional source objects still invalidate the complete literal scope.
    """
    verbs = r"查找|检索|读取|查看|复述|还原"
    reference = r"(?:上述|前述|这些|这两份|这两条|该|指定的)"
    speech = r"(?:完整|全部)?(?:用户)?(?:原话|原始发言)(?:记录)?"
    direct = rf"(?:{verbs})\s*(?:{reference})?{speech}\s*(?:仅仅|只是|仅|只)(?:用于|用作|作为|为了)"
    past_reference = (rf"(?:{verbs})(?:过|到)?的\s*(?:{reference})?{speech}\s*"
                      r"(?:仅仅|只是|仅|只)?(?:用作|作为)(?:核对|对照|比较|判断|分析|审查)(?:依据|材料)")
    purpose = re.compile(rf"{direct}|{past_reference}")
    masked = list(view)
    for match in purpose.finditer(view):
        # A nominal explanation cannot waive an explicitly new or negated
        # action, even if it names a previously read source.
        clause_start = max(view.rfind(char, 0, match.start()) for char in '。；;\n') + 1
        prefix = view[clause_start:match.start()]
        if re.search(r"另外|再|还要|还需|顺便|此外|不要|别|禁止|并且|其他|另一|新的|不同", prefix):
            continue
        masked[match.start():match.end()] = ' ' * (match.end() - match.start())
    remainder = ''.join(masked)
    if re.search(rf"(?:{verbs})[^。；;\n]*(?:原话|原始|记录)", remainder):
        return True
    # A connector alone is not a new retrieval: conditions, output fields and
    # comparison instructions can refer to the already declared evidence.
    referred_object = rf"{reference}(?:完整|全部)?(?:用户)?(?:原话|原始发言|原始记录|记录|资料|目录|文档|版本|历史)(?:记录)?"
    remainder = re.sub(referred_object, lambda m: ' ' * len(m.group()), remainder)
    addition = r"(?:另外|同时|以及|还要|还需|并且|顺便|此外|也要)"
    source_object = r"(?:原话|原始(?:资料|发言|记录)|记录|资料|档案|文档|目录|版本|历史)"
    action = r"(?:核对|对照|比较|检查|补充|列出|提供|找出|查询|搜索)"
    return bool(re.search(rf"{addition}[^。；;\n]*{action}[^。；;\n]*{source_object}", remainder)
                or re.search(rf"{addition}[^。；;\n]*(?:其他|另一|另一个|新的|不同|全部历史|所有历史)[^。；;\n]*{source_object}", remainder))



def literal_source_read_groups(query):
    """Closed literal read declarations, as a union of same-record groups.

    A conjunction requires every fragment in one original speech record.
    Disjunctions and separately closed reads are unions. Mixed operators have
    no guessed precedence; unresolved or quoted commands retain sparse recall.
    """
    from character.quoted_erasure_authority import masked_quotes

    if not isinstance(query, str):
        return ()
    try:
        view, quotes = masked_quotes(query)
    except ValueError:
        return ()
    groups, position, index = [], 0, 0
    prefix = (r"\s*(?:(?:再|另|同时|另外)?(?:请)?|请(?:再|另|同时|另外)?)"
              r"(?:查找|检索|读取|查看)\s*(?:包含|含有)\s*")
    suffix = r"\s*的\s*(?:完整|全部)?(?:用户)?(?:原话|原始发言)(?:记录)?\s*(?:[。；;\n]|$)"
    while index < len(quotes):
        a, _, _ = quotes[index]
        if not re.fullmatch(prefix, view[position:a]):
            break
        members, operator = [], None
        while index < len(quotes):
            _, z, fragment = quotes[index]
            if (not fragment.strip() or "\n" in fragment or not terms(fragment)
                    or any(char in fragment for char in '“”‘’「」『』"')):
                return ()
            members.append(fragment)
            index += 1
            read = re.match(suffix, view[z:])
            if read:
                position = z + read.end()
                members = tuple(dict.fromkeys(members))
                groups.extend((member,) for member in members) if operator == 'or' else groups.append(members)
                break
            if index >= len(quotes):
                return ()
            connector = view[z:quotes[index][0]].strip()
            next_operator = ('and' if connector in {'和', '与', '及', '并且'} else
                             'or' if connector in {'或', '或者'} else None)
            if next_operator is None or (operator is not None and next_operator != operator):
                return ()
            operator = next_operator
        else:
            return ()
    remainder = view[position:]
    if not groups or unresolved_source_read_tail(remainder):
        return ()
    return tuple(dict.fromkeys(groups))



def resolve_source_read_plan(query):
    from knowledge.source_read_plan import source_read_plan

    return source_read_plan(query, terms, unresolved_source_read_tail)


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
    read_plan = resolve_source_read_plan(query)
    groups = read_plan.atom_groups
    project_tags = () if groups else literal_project_source_terms(query)
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
    if groups:
        fragments = read_plan.selectors
        names = {fragment: f"source_fragment{index}" for index, fragment in enumerate(fragments)}
        params.update({names[fragment]: fragment.value for fragment in fragments})
        def expression(fragment):
            name = names[fragment]
            if fragment.match_mode == 'identifier_token':
                if dialect == 'sqlite':
                    return f"source_identifier_match(:{name}, eligible.body) = 1"
                from knowledge.source_read_plan import identifier_pattern
                pattern_name = name + '_pattern'
                params[pattern_name] = identifier_pattern(fragment.value)
                return f"eligible.body ~ :{pattern_name}"
            return (f"instr(eligible.body, :{name}) > 0" if dialect == "sqlite" else
                    f"POSITION(:{name} IN eligible.body) > 0")
        fragment_filter = "AND (" + " OR ".join(
            "(" + " AND ".join(expression(fragment) for fragment in group) + ")"
            for group in groups) + ") "
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
