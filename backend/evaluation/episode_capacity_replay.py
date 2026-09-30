"""Cross persona context with evidence packaging to localize reading failures.

Frozen no-LoRA diagnostic only. Plain controls are not persona products and
must not be promoted just because they answer factual questions more often.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from evaluation.episode_role_replay import role_views
from evaluation.episode_subject_audit import complete
from evaluation.replay_evidence_ablation import replay_parameters


def capacity_views(trace, selection):
    call = trace['model_calls'][0]
    framed = role_views(call['messages'], selection, trace['case_id'])['history_pairs']
    query = trace['message']
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Missing original query')
    natural = []
    for row in json.loads(selection['packet']):
        natural.append(dict(role='user', content=row['message']))
        if row['reply']:
            natural.append(dict(role='assistant', content=row['reply']))
    natural.append(dict(role='user', content=query))
    return dict(framed_persona=framed, framed_plain=framed[1:],
                natural_persona=[dict(framed[0]), *natural], natural_plain=natural)


def without_ballast_views(trace, selection, case):
    """Remove only exact synthetic ballast inserted by seed_cases, not retrieval.

    Fixture episode membership was fixed before inference. Never use a gold
    answer, semantic relevance guess, or model output to select these turns.
    """
    views = capacity_views(trace, selection)
    rows = json.loads(selection['packet'])
    core = case['episodes']
    owner = 'episode-chain-' + case['id']
    if owner != trace['case_id'] or case['query'] != trace['message'] or len(rows) != len(core) + 30:
        raise ValueError('Not the expected complete seeded fixture')
    for actual, expected in zip(rows, core):
        if (any(actual[key] != expected[key] for key in ('message', 'reply', 'timestamp'))
                or actual['session'] != owner + '-' + expected['session']):
            raise ValueError('Core fixture evidence differs')
    for index, actual in enumerate(rows[len(core):]):
        if (actual['message'] != f'这一页的段落编号是{index}。'
                or actual['reply'] != '收到这一页的编号。'
                or actual['timestamp'] != f'2026-09-25T12:{index:02d}:00'
                or actual['session'] != owner):
            raise ValueError('Refusing to drop non-ballast history')
    core_frames = sum(1 + bool(row['reply']) for row in core)
    plain = [*views['natural_plain'][:core_frames], views['natural_plain'][-1]]
    return dict(core_persona=[views['natural_persona'][0], *plain], core_plain=plain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--selections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--without-ballast-suite', type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError('Expected bounded repeats')
    traces = [json.loads(s) for s in args.traces.read_text(encoding='utf-8').splitlines()]
    selections = [json.loads(s) for s in args.selections.read_text(encoding='utf-8').splitlines()]
    if len(traces) != len(selections):
        raise ValueError('Mismatched trace/selection counts')
    # Validate all input pairs before the first billable inference call.
    prepared = []
    suite = ({'episode-chain-' + row['id']: row for row in
              json.loads(args.without_ballast_suite.read_text(encoding='utf-8'))}
             if args.without_ballast_suite else None)
    for trace, selection in zip(traces, selections):
        call = trace['model_calls'][0]
        if call['parameters'].get('lora_name') not in {None, '', 'default'}:
            raise ValueError('Requires no-LoRA inputs')
        views = capacity_views(trace, selection)
        if suite is not None:
            views = without_ballast_views(trace, selection, suite[trace['case_id']])
        params, assumed = replay_parameters(call, trace['generation']['plan']['generation'])
        prepared.append((trace, selection, views, params, assumed))
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.traces, args.selections, Path(__file__),
               Path(role_views.__code__.co_filename), Path(replay_parameters.__code__.co_filename)]
    if args.without_ballast_suite:
        sources.append(args.without_ballast_suite)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        runtime_enabled=False, model=args.model, repeats=args.repeats,
        planned_calls=sum(len(item[2]) for item in prepared)*args.repeats, model_writes=False,
        exact_synthetic_ballast_removed=suite is not None,
        http_application_e2e=False, client_truncation=False,
        caveat='Natural views omit timestamps and source IDs; not a temporal-reasoning replacement.',
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2), encoding='utf-8')
    for trace, selection, views, params, assumed in prepared:
        for seed in range(args.repeats):
            for name in list(views)[::1 if seed % 2 == 0 else -1]:
                body = dict(params, model=args.model, messages=views[name], seed=seed)
                body['chat_template_kwargs'] = {'enable_thinking': body.pop('enable_thinking')}
                started = time.monotonic()
                response = complete(args.endpoint, body)
                record = dict(case=trace['case_id'], variant=name, seed=seed,
                    request=body, response=response, seconds=time.monotonic()-started,
                    source_ids=selection['source_ids'], assumed_parameters=assumed,
                    interpreted_quality='requires_manual_review_not_automatic_pass')
                with (args.output / 'replays.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(dict(case=record['case'], variant=name, seed=seed)), flush=True)


if __name__ == '__main__':
    main()
