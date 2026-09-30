"""Offline audit of whole-reply preference exemptions, not an entailment judge.

Forcing the detector on identifies review candidates, NOT hallucinations.
Keep actual admitted evidence next to each candidate for independent review.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

from character.memory_extractor import extract_memories
from character.output_guard import UNSUPPORTED_USER_FACT, ReplyGuard, validate_reply


def audit_trace(row: dict) -> dict:
    prepared = row.get('prepared') or {}
    calls = row.get('model_calls') or []
    raw_guard = prepared.get('reply_guard')
    result = {key: row.get(key) for key in ('case', 'case_id', 'index', 'turn', 'phase', 'message')}
    if (not calls or not isinstance(raw_guard, dict)
            or not isinstance(raw_guard.get('forbid_unsupported_user_fact'), bool)):
        return dict(result, status='not_replayable')
    if set(raw_guard) - {field.name for field in fields(ReplyGuard)}:
        return dict(result, status='incompatible_guard_schema')
    reply = calls[0].get('reply')
    if not isinstance(reply, str):
        return dict(result, status='missing_first_reply')
    guard = ReplyGuard(**raw_guard)
    compiled = prepared.get('compiled') or {}
    # Prepared history is the input view; final SQLite state includes this
    # turn's later write and must never become evidence for its own answer.
    user_texts = [h.get('content', '') for h in prepared.get('history', []) if h.get('role') == 'user']
    user_texts.append(row.get('message') or '')
    extracted = [dict(memory_key=m.memory_key, content=m.content, evidence=m.evidence)
                 for text in user_texts for m in extract_memories(text)
                 if m.memory_key.startswith('preference_')]
    detected = UNSUPPORTED_USER_FACT in validate_reply(reply, replace(guard, forbid_unsupported_user_fact=True))
    source = compiled.get('episodic_reference_context') or ''
    reasons = []
    if compiled.get('used_memory_ids'):
        reasons.append('any_admitted_memory')
    if extracted:
        reasons.append('any_asserted_preference')
    return dict(result, status='audited', first_reply=reply,
        gate_disabled=not guard.forbid_unsupported_user_fact,
        detector_matches=detected,
        review_candidate=not guard.forbid_unsupported_user_fact and detected,
        possible_exemption_reasons=reasons,
        reason_verified=False, semantic_verdict='not_evaluated',
        prepared_user_texts=user_texts, extracted_user_preferences=extracted,
        model_user_messages=[m['content'] for m in (calls[0].get('request') or {}).get('messages', [])
                             if m.get('role') == 'user'],
        used_memory_ids=compiled.get('used_memory_ids', []),
        memory_packets=compiled.get('memory_packets', []),
        reference_context=compiled.get('reference_context', ''),
        original_source_packet=source,
        recorded_violations=(row.get('generation') or {}).get('guard_violations'),
        recorded_final_reply=(row.get('generation') or {}).get('reply'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    with args.output.open('x', encoding='utf-8') as out:
        for path in args.trace:
            for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                item = dict(source=str(path), source_line=index, **audit_trace(json.loads(line)))
                counts[item['status']] += 1
                for key in ('gate_disabled', 'detector_matches', 'review_candidate'):
                    counts[key] += bool(item.get(key))
                out.write(json.dumps(item, ensure_ascii=False)+'\n')
    print(json.dumps(dict(counts=counts, new_model_calls=0, semantic_verdict='not_evaluated')))


if __name__ == '__main__':
    main()
