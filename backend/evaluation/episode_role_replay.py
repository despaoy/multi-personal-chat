"""Frozen-input source-role ablation, not production history admission.

Requires no existing history: cannot safely deduplicate/reorder overlapping
retrieved and recent turns without source IDs on both sides. Assistant words
remain historical outputs, never certified user facts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from html import escape
from pathlib import Path

from evaluation.episode_subject_audit import complete
from evaluation.episodic_packet_replay import replace_packet
from evaluation.episodic_reader_replay import evidence_view
from evaluation.replay_evidence_ablation import replay_parameters


def role_views(messages, selection, case_id):
    if len(messages) != 2 or [row['role'] for row in messages] != ['system', 'user']:
        raise ValueError('Requires no-history system/user input')
    if selection.get('history') or not selection.get('applied'):
        raise ValueError('Requires applied evidence without overlapping recent history')
    rows = json.loads(selection['packet'])
    if not rows or [row['source_id'] for row in rows] != selection['source_ids']:
        raise ValueError('Incomplete source rows')
    scope = rows[0]['scope']
    if (len(scope) != 6 or scope[2] != case_id or any(row['scope'] != scope for row in rows)
            or len(set(selection['source_ids'])) != len(rows)):
        raise ValueError('Mixed or duplicate source scope')
    if rows != sorted(rows, key=lambda row: (row['timestamp'], row['source_id'])):
        raise ValueError('Source chronology is not ordered')
    packet = selection['injected_packet']
    if evidence_view(selection['packet'], 'user_only') != packet:
        raise ValueError('Injected evidence does not reconstruct source users')
    block = ('<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
             + escape(packet, quote=False) + '\n</dialogue_evidence>')
    if messages[-1]['content'].count(block) != 1:
        raise ValueError('Expected exact separate historical evidence channel')
    final = dict(messages[-1], content=messages[-1]['content'].replace(block, ''))
    users, pairs = [], []
    for row in rows:
        if not isinstance(row['message'], str) or not row['message'].strip():
            raise ValueError('Cannot fabricate missing user turns')
        if not isinstance(row['reply'], str):
            raise ValueError('Invalid assistant text')
        # Same source metadata and exact utterance in flat/pair/user views.
        metadata = {key: row[key] for key in ('source_id', 'session', 'timestamp')}
        content = json.dumps(metadata, ensure_ascii=False, separators=(',', ':')) + '\n' + row['message']
        user = dict(role='user', content=content)
        users.append(user)
        pairs.append(user)
        if row['reply']:
            pairs.append(dict(role='assistant', content=row['reply']))
    flat = '\n\n'.join(row['role'] + ':\n' + row['content'] for row in pairs)
    return dict(original=messages, flat_pairs=replace_packet(messages, packet, flat),
        history_users=[dict(messages[0]), *users, final],
        history_pairs=[dict(messages[0]), *pairs, final])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--selections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError('Expected bounded repeats')
    traces = [json.loads(line) for line in args.traces.read_text(encoding='utf-8').splitlines()]
    selections = [json.loads(line) for line in args.selections.read_text(encoding='utf-8').splitlines()]
    if len(traces) != len(selections):
        raise ValueError('Mismatched traces and selections')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.traces, args.selections, Path(__file__))}
    (args.output / 'manifest.json').write_text(json.dumps(dict(sources=hashes,
        runtime_enabled=False, guard_executed=False, memory_writes=False,
        chronological_metadata_retained=True, client_truncation=False,
        runtime_budgeter_executed=False), indent=2), encoding='utf-8')
    for trace, selection in zip(traces, selections):
        calls = trace['model_calls']
        if not calls or calls[0]['parameters'].get('lora_name') not in {None, '', 'default'}:
            raise ValueError('Requires actual no-LoRA call')
        # Replays the original first attempt, never a guard-retry prompt.
        call = calls[0]
        views = role_views(call['messages'], selection, trace['case_id'])
        params, assumed = replay_parameters(call, trace['generation']['plan']['generation'])
        for seed in range(args.repeats):
            names = list(views) if seed % 2 == 0 else list(views)[::-1]
            for name in names:
                body = dict(params, model=args.model, messages=views[name], seed=seed)
                body['chat_template_kwargs'] = {'enable_thinking': body.pop('enable_thinking')}
                started = time.monotonic()
                response = complete(args.endpoint, body)
                record = dict(case=trace['case_id'], variant=name, seed=seed, request=body,
                    response=response, seconds=time.monotonic()-started, assumed_parameters=assumed,
                    source_ids=selection['source_ids'], original_chain_calls=len(calls),
                    interpreted_quality='not_automatically_scored')
                with (args.output / 'replays.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(dict(case=record['case'], variant=name, seed=seed)), flush=True)


if __name__ == '__main__':
    main()
