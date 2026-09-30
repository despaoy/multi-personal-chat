"""Lossless user-evidence metadata packing and frozen-input model comparison.

Offline diagnosis only. No runtime registration, guard execution or writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from html import escape
from pathlib import Path
from urllib.request import Request, urlopen

from evaluation.replay_evidence_ablation import replay_parameters


def compact_packet(packet):
    rows = json.loads(packet)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty user evidence')
    keys = {'source_id', 'scope', 'session', 'timestamp', 'message'}
    if any(set(row) != keys for row in rows):
        raise ValueError('Unexpected evidence schema')
    scope = rows[0]['scope']
    if any(row['scope'] != scope for row in rows):
        raise ValueError('Mixed evidence owners')
    sessions = list(dict.fromkeys(row['session'] for row in rows))
    compact = dict(scope=scope, sessions=sessions, records=[dict(source_id=row['source_id'],
        session_index=sessions.index(row['session']), timestamp=row['timestamp'],
        message=row['message']) for row in rows])
    return json.dumps(compact, ensure_ascii=False, separators=(',', ':'))


def expand_packet(packet):
    data = json.loads(packet)
    return [dict(source_id=row['source_id'], scope=data['scope'],
                 session=data['sessions'][row['session_index']],
                 timestamp=row['timestamp'], message=row['message']) for row in data['records']]


def provenance_packet(packet):
    """Expose known dialogue speaker, never infer whom the utterance describes.

    Input is the scoped SQLite user_only packet, not arbitrary quoted text.
    This is still an experimental representation, not a semantic extractor.
    """
    data = json.loads(compact_packet(packet))
    scope = data['scope']
    if not isinstance(scope, list) or len(scope) != 6 or not all(scope[:4]):
        raise ValueError('Expected authenticated SQLite episode scope')
    data['provenance'] = dict(source_kind='historical_user_utterances',
        speaker_role='user', speaker_id=scope[2], conversation_character_id=scope[3],
        described_subject='not_resolved', current_validity='not_resolved')
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def replace_packet(messages, old_packet, new_packet):
    # Match the escaped serialized evidence actually recorded in the input.
    old, new = escape(old_packet, quote=False), escape(new_packet, quote=False)
    if sum(m['content'].count(old) for m in messages) != 1:
        raise ValueError('Expected exactly one complete packet in model input')
    if messages[-1]['role'] != 'user' or messages[-1]['content'].count(old) != 1:
        raise ValueError('Evidence must be in the final user input')
    result = [dict(m) for m in messages]
    result[-1]['content'] = result[-1]['content'].replace(old, new)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--selections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--compare-provenance', action='store_true')
    args = parser.parse_args()
    traces = [json.loads(line) for line in args.traces.read_text(encoding='utf-8').splitlines()]
    selections = [json.loads(line) for line in args.selections.read_text(encoding='utf-8').splitlines()]
    if len(traces) != len(selections):
        raise ValueError('Trace/selection count mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (args.traces, args.selections, Path(__file__))}
    (args.output / 'sources.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    for trace, selection in zip(traces, selections):
        calls = trace['model_calls']
        if len(calls) != 1 or not selection['applied']:
            raise ValueError('Expected one actual call with applied evidence')
        call = calls[0]
        if call['parameters'].get('lora_name') not in {None, '', 'default'}:
            raise ValueError('No-LoRA only')
        packet = selection['injected_packet']
        compact = compact_packet(packet)
        assert expand_packet(compact) == json.loads(packet)
        views = dict(original=call['messages'], compact=replace_packet(call['messages'], packet, compact))
        names = ['original', 'compact']
        if args.compare_provenance:
            views = dict(compact=views['compact'], provenance=replace_packet(
                call['messages'], packet, provenance_packet(packet)))
            names = ['compact', 'provenance']
        params, assumed = replay_parameters(call, trace['generation']['plan']['generation'])
        for seed in range(args.repeats):
            for name in (names if seed % 2 == 0 else names[::-1]):
                body = dict(params, model=args.model, messages=views[name], seed=seed)
                body['chat_template_kwargs'] = {'enable_thinking': body.pop('enable_thinking')}
                started = time.monotonic()
                headers = {'Content-Type': 'application/json'}
                if os.getenv('VLLM_API_KEY'):
                    headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
                req = Request(args.endpoint, data=json.dumps(body).encode(),
                              headers=headers)
                with urlopen(req, timeout=180) as response:
                    result = json.load(response)
                record = dict(case=trace['case_id'], variant=name, seed=seed, request=body,
                    response=result, seconds=time.monotonic()-started, assumed_parameters=assumed,
                    original_chars=len(packet), compact_chars=len(compact), guard_executed=False)
                with (args.output / 'replays.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(dict(case=trace['case_id'], variant=name, seed=seed)), flush=True)


if __name__ == '__main__':
    main()
