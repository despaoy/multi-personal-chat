"""Paired inference-mode diagnostic with identical frozen inputs and budgets.

No persona/prompt edits, retrieval, guard, or memory writes. Final text is kept
separate from model scratchpad; incomplete reasoning is not an answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from evaluation.episode_subject_audit import complete
from evaluation.replay_evidence_ablation import replay_parameters


def final_view(response):
    choice = response['choices'][0]
    content = choice['message'].get('content') or ''
    # This experiment only accepts complete final responses. A length stop can
    # contain a valid prefix but is not comparable to a finished answer.
    if choice['finish_reason'] != 'stop':
        return dict(text='', status='incomplete_generation')
    if '<think>' in content.lower():
        if content.lower().count('<think>') != content.lower().count('</think>'):
            return dict(text='', status='incomplete_reasoning')
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.I | re.S)
    elif '</think>' in content.lower():
        # Some templates already include the opener in the generation prompt.
        content = re.split(r'</think>', content, flags=re.I)[-1]
    return dict(text=content.strip(), status='complete' if content.strip() else 'empty_final')


def paired_requests(trace, model, seed):
    call = trace['model_calls'][0]
    if call['parameters'].get('lora_name') not in {None, '', 'default'}:
        raise ValueError('No-LoRA only')
    params, assumed = replay_parameters(call, trace['generation']['plan']['generation'])
    params.pop('enable_thinking')
    return {name: (dict(params, model=model, messages=call['messages'], seed=seed,
                       chat_template_kwargs={'enable_thinking': enabled}), assumed)
            for name, enabled in (('disabled', False), ('enabled', True))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        raise ValueError('Invalid repeats')
    traces = [json.loads(line) for path in args.traces
              for line in path.read_text(encoding='utf-8').splitlines()]
    if len({(row['case_id'], row['turn']) for row in traces}) != len(traces):
        raise ValueError('Duplicate case/turn')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (*args.traces, Path(__file__))}
    (args.output / 'manifest.json').write_text(json.dumps(dict(sources=hashes,
        runtime_enabled=False, memory_writes=False, guard_executed=False,
        inputs_and_total_generation_budget_identical=True), indent=2), encoding='utf-8')
    for index, trace in enumerate(traces):
        for seed in range(args.repeats):
            variants = paired_requests(trace, args.model, seed)
            order = list(variants) if (index + seed) % 2 == 0 else list(variants)[::-1]
            for name in order:
                body, assumed = variants[name]
                started = time.monotonic()
                response = complete(args.endpoint, body)
                record = dict(case=trace['case_id'], variant=name, seed=seed,
                    request=body, response=response, final=final_view(response),
                    seconds=time.monotonic()-started, assumed_parameters=assumed)
                with (args.output / 'replays.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                # Do not dump private scratchpad to console or interpret it as evidence.
                print(json.dumps(dict(case=record['case'], variant=name,
                    seconds=record['seconds'], status=record['final']['status'])), flush=True)


if __name__ == '__main__':
    main()
