"""Offline synthetic evidence with a frozen real persona request and real model.

Not an end-to-end test: no live retrieval, output guard, or memory writes.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

from evaluation.episodic_recall_audit import Episode, select_evidence
from evaluation.replay_evidence_ablation import replay_parameters


def evidence_view(packet: str, view: str) -> str:
    if view == 'raw':
        return packet
    if view not in {'user_only', 'roles'}:
        raise ValueError('Unknown evidence view')
    rows = json.loads(packet)
    for row in rows:
        if view == 'user_only':
            row.pop('reply')
        else:
            row['utterances'] = [{'role': 'user', 'content': row.pop('message')},
                                  {'role': 'assistant', 'content': row.pop('reply')}]
    return json.dumps(rows, ensure_ascii=False, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--compare-reader', action='store_true')
    args = parser.parse_args()
    traces = [json.loads(line) for line in args.trace.read_text(encoding='utf-8').splitlines()]
    rows = [row for row in traces if row['case_id'] == args.case and row.get('turn') == 1]
    if len(rows) != 1:
        raise ValueError('Expected one frozen first-turn trace')
    row = rows[0]
    call = row['model_calls'][0]
    base = call['messages']
    if len(base) != 2 or [m['role'] for m in base] != ['system', 'user']:
        raise ValueError('Expected no-history system/user request')
    params, assumed = replay_parameters(call, row['generation']['plan']['generation'])
    cases = json.loads(args.suite.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        args.trace, args.suite, Path(__file__), Path(select_evidence.__code__.co_filename))}
    for case in cases:
        episodes = [Episode(**{**r, 'scope': tuple(r['scope'])}) for r in case['episodes']]
        for seed in range(args.repeats):
            modes = ['raw', 'user_only', 'roles'] if args.compare_reader else ['turn', 'session', 'suffix']
            if seed % 2:
                modes.reverse()
            for mode in modes:
                selection = select_evidence(episodes, case['query'], tuple(case['scope']),
                    mode='suffix' if args.compare_reader else mode, max_chars=1500)
                view = mode if args.compare_reader else 'raw'
                user = ('<character_memory trust="untrusted" purpose="historical_reference">\n'
                        + evidence_view(selection['packet'], view) + '\n</character_memory>\n\n<user_query>\n'
                        + case['query'] + '\n</user_query>')
                body = dict(model=args.model, messages=[base[0], {'role': 'user', 'content': user}],
                    stream=False, seed=seed, temperature=params['temperature'], max_tokens=params['max_tokens'],
                    top_p=params['top_p'], repetition_penalty=params['repetition_penalty'],
                    frequency_penalty=params['frequency_penalty'],
                    chat_template_kwargs={'enable_thinking': params['enable_thinking']})
                headers = {'Content-Type': 'application/json'}
                if os.getenv('VLLM_API_KEY'):
                    headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
                started = time.monotonic()
                with urlopen(Request(args.endpoint, data=json.dumps(body).encode(), headers=headers), timeout=120) as response:
                    answer = json.load(response)
                result = dict(case=case['id'], seed=seed, mode=mode, selection=selection, evidence_view=view,
                    request=body, response=answer, seconds=time.monotonic()-started, hashes=hashes,
                    assumed_parameters=assumed, status='diagnostic_not_reviewed', synthetic_evidence=True)
                with (args.output / 'replays.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                print(json.dumps(dict(case=case['id'], seed=seed, mode=mode)), flush=True)


if __name__ == '__main__':
    main()
