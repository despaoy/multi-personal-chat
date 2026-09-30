"""Replay frozen first model replies; never invoke a model or rewrite history.

This compares guard decisions, not overall answer quality. The stored first-pass
violations are the baseline; later retry/fallback text must not replace them.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from character.output_guard import ReplyGuard, ground_reply_guard, validate_reply


def replay_trace(row: dict) -> dict:
    result = {key: row.get(key) for key in ('case_id', 'turn', 'message')}
    calls = row.get('model_calls')
    generation = row.get('generation') or {}
    guard = (row.get('prepared') or {}).get('reply_guard')
    if not calls or not isinstance(guard, dict) or 'guard_violations' not in generation:
        return {**result, 'status': 'not_replayable'}
    if set(guard) - {field.name for field in fields(ReplyGuard)}:
        return {**result, 'status': 'incompatible_guard_schema'}
    reply = calls[0].get('reply')
    if not isinstance(reply, str):
        return {**result, 'status': 'missing_first_reply'}
    before = tuple(generation['guard_violations'])
    effective_guard = ReplyGuard(**guard)
    retrieval = (generation.get('plan') or {}).get('retrieval') or {}
    evidence = retrieval.get('evidence') or ''
    if retrieval.get('status') == 'ok' and evidence.strip():
        effective_guard = ground_reply_guard(effective_guard, evidence)
    after = validate_reply(reply, effective_guard)
    result.update(status='replayed', reply=reply, before=before, after=after, changed=before != after)
    if generation.get('guard_retried'):
        if len(calls) != 2 or 'guard_post_retry_violations' not in generation:
            return {**result, 'status': 'incomplete_retry_trace'}
        retry_before = tuple(generation['guard_post_retry_violations'])
        retry_after = validate_reply(calls[-1]['reply'], effective_guard)
        result.update(retry_before=retry_before, retry_after=retry_after,
                      changed=result['changed'] or retry_before != retry_after)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.trace:
        rows.extend({'source': str(path), **replay_trace(json.loads(line))}
                    for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    with args.output.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    changed = [row for row in rows if row.get('changed')]
    print(json.dumps(dict(total=len(rows), replayed=sum(row['status'] == 'replayed' for row in rows),
                          changed=changed), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
