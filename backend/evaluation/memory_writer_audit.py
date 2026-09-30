"""Replay recorded writer admission, without a model or storage writes.

Counts describe syntax and current admission, never factual correctness.
The recorded writer payload supplies the exact old-memory whitelist/history.
"""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from character.memory_llm import _extract_json, parse_llm_proposals


def audit_call(call):
    payload = json.loads(call['messages'][-1]['content'])
    result = dict(source=payload['current_user_message'], candidates=[], accepted=[])
    try:
        candidates = _extract_json(call['output'])['memories']
        if not isinstance(candidates, list):
            raise ValueError('memories is not an array')
        result['candidates'] = candidates
    except (ValueError, TypeError, KeyError) as exc:
        return dict(result, status='format_failed', error=str(exc))
    proposals = parse_llm_proposals(call['output'], source_message=result['source'],
        history=tuple(payload.get('recent_history', ())),
        existing_memories=tuple(payload.get('existing_memories', ())),
        feedback_target_ids=tuple(payload.get('feedback_target_ids', ())),
        confidence_threshold=payload.get('confidence_threshold', 0.85))
    result['accepted'] = [asdict(item) for item in proposals]
    return dict(result, status=('empty_candidates' if not candidates else
                               'admitted' if proposals else 'none_admitted'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    # Exclusive creation prevents overwriting another evaluation.
    with args.output.open('x', encoding='utf-8') as out:
        for path in args.trace:
            for row in map(json.loads, path.read_text(encoding='utf-8').splitlines()):
                for index, call in enumerate(row.get('writer_calls', ())):
                    result = audit_call(call)
                    result.update(trace=str(path), case=row['case'], turn=row['index'], call=index)
                    counts[result['status']] += 1
                    out.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(json.dumps(dict(writer_calls=sum(counts.values()), statuses=dict(counts),
                         fresh_model_calls=0, database_writes=0)))


if __name__ == '__main__':
    main()
