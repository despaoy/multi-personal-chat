"""Keep source-only speech fragments when a linked preference is erased.

These are literal observations with original clocks and source identities, not
new facts or replacements for the complete source. The caller holds its owner
lock and commits the plans with existing claim deletion and source revocation.
"""

import hashlib
import json
import re
from datetime import datetime, timezone

from db.memory_claim_guard import MemoryClaimConflict
from db.memory_source import source_scope, utc_stamp

_INTRO = re.compile(
    r"(?:逐字)?(?:引用|复述|转述)我(?:本人|自己)?(?:以前|此前|先前|过去|之前)的(?:一句话|原话|发言)\s*[:：]\s*"
)
_PAIRS = {"“": "”", "「": "」", "『": "』", '"': '"'}
_QUALIFIER = r"(?:[，,]\s*这是我(?:本人)?(?:明确(?:且|而且))?(?:长期)?的(?:个人)?偏好)?"


def historical_fragments(body, values, outside_quotes):
    """Only explicit own-history quotes of a complete literal preference.

    Nothing outside the quote is inferred to be a course fact. Unknown quote
    syntax, extra predicates, role corrections or remaining copies conflict;
    a broad old citation cannot authorize deleting adjacent source material.
    """
    if re.search(
        r"(?:不是|并非|不属于)我(?:本人|自己)?(?:的|说)|(?:实际|其实|更正).{0,12}(?:朋友|同事|别人|不是我)", body
    ):
        raise MemoryClaimConflict("historical quotation ownership is unresolved")
    cuts = []
    covered = set()
    for intro in _INTRO.finditer(body):
        start = intro.end()
        if not outside_quotes(body, intro.start()) or start == len(body) or body[start] not in _PAIRS:
            continue
        close = _PAIRS[body[start]]
        end = body.find(close, start + 1)
        if end < 0:
            raise MemoryClaimConflict("historical quotation is incomplete")
        quote = body[start + 1 : end]
        matched = [
            value
            for value in values
            if re.fullmatch(
                r"\s*我(?:本人)?(?:喜欢|偏好|偏爱)" + re.escape(value) + _QUALIFIER + r"[。！？!?；;.]?\s*", quote
            )
        ]
        if len(matched) != 1:
            continue
        if not outside_quotes(body, end + 1):
            raise MemoryClaimConflict("historical quotation boundaries are unresolved")
        cuts.append((start, end + 1))
        covered.add(matched[0])
    relevant = {value for value in values if value in body}
    if not cuts or covered != relevant:
        raise MemoryClaimConflict("source-only preference copy cannot be separated")
    fragments, spans = [], []
    cursor = 0
    for start, end in [*sorted(set(cuts)), (len(body), len(body))]:
        if start < cursor:
            raise MemoryClaimConflict("historical quotation spans overlap")
        if cursor < start and body[cursor:start].strip():
            fragments.append(body[cursor:start])
            spans.append([cursor, start])
        cursor = end
    if not fragments or any(value in fragment for value in values for fragment in fragments):
        raise MemoryClaimConflict("erased preference remains in source fragments")
    return fragments, spans


def observation_plan(scope, erased, linked_sources, preference_cuts, outside_quotes):
    """Plan exact-scope unlinked observations before advancing the owner fence.

    Actual erased rows and their genuine linked recorded sources establish the
    literal preference. Previously fenced sources never become eligible again.
    The candidate cap is a safety bound; overflow is an atomic conflict.
    """
    values = set()
    for entry in linked_sources.values():
        source = entry["source"]
        if source.get("state") != "recorded" or not isinstance(source.get("body"), str):
            continue
        for identity in entry["erased_ids"]:
            record = erased[identity]
            if not str(record.get("memory_key") or "").startswith("preference_"):
                continue
            try:
                preference_cuts(source["body"], record)
            except MemoryClaimConflict:
                continue
            values.add(record["memory_key"].removeprefix("preference_"))
    if not values:
        return []
    fields = json.loads(scope["scope_key"])
    if not isinstance(fields, list) or len(fields) != 6 or source_scope(*fields) != scope:
        raise MemoryClaimConflict("source fragment requires an exact original scope")
    candidates = {}
    for value in sorted(values):
        needle = "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        rows = yield (
            "SELECT s.source_key,s.source_message_id,s.observed_at,s.body FROM memory_sources s "
            "WHERE s.owner_key=:owner_key AND s.scope_key=:scope_key AND s.state='recorded' "
            "AND s.observed_at > COALESCE((SELECT revoked_before FROM memory_source_fences "
            "WHERE owner_key=:owner_key),'') AND s.body LIKE :needle ESCAPE '\\' "
            "AND NOT EXISTS (SELECT 1 FROM memory_source_links l WHERE l.source_key=s.source_key) "
            "ORDER BY s.observed_at,s.source_key LIMIT 201",
            dict(scope, needle=needle),
        )
        for row in rows:
            candidates[row["source_key"]] = row
        if len(candidates) > 200:
            raise MemoryClaimConflict("source fragment candidate bound exceeded")
    plans = []
    now = utc_stamp(datetime.now(timezone.utc))
    for source in candidates.values():
        fragments, spans = historical_fragments(source["body"], values, outside_quotes)
        try:
            # Preserve the exact actual timestamp, never substitute erasure time.
            utc_stamp(datetime.fromisoformat(source["observed_at"]))
        except (ValueError, TypeError) as exc:
            raise MemoryClaimConflict("source fragment clock is unavailable") from exc
        metadata = dict(
            content_semantics="quoted_source",
            speaker_role="user",
            described_subject="not_resolved",
            attributed_to="user",
            qualifiers={},
            temporal_provenance=dict(version=1, producer="source_erasure_projection", validity_authority="unspecified"),
            erasure_evidence_projection=dict(
                version=1,
                kind="original_source_fragments",
                sources=[dict(source_message_id=source["source_message_id"], spans=spans)],
                complete_original_source=False,
            ),
        )
        digest = hashlib.sha256(
            json.dumps([source["source_key"], sorted(erased)], ensure_ascii=False).encode()
        ).hexdigest()
        plans.append(
            dict(
                zip(
                    ["character_id", "platform", "adapter", "sender_id", "conversation_type", "conversation_id"],
                    fields,
                    strict=True,
                ),
                memory_key="source_fragment_" + digest,
                content="用户原话保留片段（仅保留内容见证据）",
                source_message_id=source["source_message_id"],
                source_key=source["source_key"],
                source_message_ids_json=json.dumps([source["source_message_id"]], ensure_ascii=False),
                evidence_json=json.dumps(fragments, ensure_ascii=False),
                metadata_json=json.dumps(metadata, ensure_ascii=False),
                observed_at=source["observed_at"],
                now=now,
            )
        )
    return plans


def apply_observations(plans):
    for params in plans:
        rows = yield (
            "INSERT INTO character_memories (character_id,platform,adapter,sender_id,conversation_type,conversation_id,"
            "scope_level,memory_type,memory_key,revision,relation_type,status,content,importance,confidence,"
            "source_message_id,source_message_ids_json,evidence_json,metadata_json,observed_at,created_at,updated_at) "
            "VALUES (:character_id,:platform,:adapter,:sender_id,:conversation_type,:conversation_id,"
            "'conversation','shared_event',:memory_key,1,'ADD','active',:content,0,1,:source_message_id,"
            ":source_message_ids_json,:evidence_json,:metadata_json,:observed_at,:now,:now) RETURNING id",
            params,
        )
        if len(rows) != 1:
            raise MemoryClaimConflict("source fragment insertion has no exact identity")
        yield (
            "INSERT INTO memory_source_links (memory_id,source_key) VALUES (:memory_id,:source_key)",
            dict(memory_id=rows[0]["id"], source_key=params["source_key"]),
        )
