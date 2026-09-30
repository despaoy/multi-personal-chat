"""Frozen real requests: ablate duplicate observation views and assistant history.

Diagnostic only: no retrieval, writes, guard, new facts or system prompt edits.
Dropping a view removes its labels and duplication together, not one word alone.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from html import escape
from pathlib import Path
from urllib.request import Request, urlopen


def request_variants(row: dict) -> dict[str, list[dict]]:
    messages = row['model_calls'][0]['request']['messages']
    compiled = row['prepared']['compiled']
    packets = compiled['memory_packets']
    if not packets or any(p['temporal_mode'] != 'observation' for p in packets):
        raise ValueError('Only an all-observation fact lane may be ablated')
    raw = json.loads(compiled['episodic_reference_context'])
    bodies = [record['text'] for record in raw['records']]
    if any(not packet['evidence'] or any(not any(quote in body for body in bodies)
            for quote in packet['evidence']) for packet in packets):
        raise ValueError('Complete original evidence must survive in the raw-source lane')
    if not any(m['role'] == 'assistant' for m in messages):
        raise ValueError('Assistant history is required for the paired ablation')
    if messages[-1]['role'] != 'user':
        raise ValueError('Expected final user query')
    raw_packet = ('<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
                  + escape(compiled['episodic_reference_context'], quote=False) + '\n</dialogue_evidence>')
    if messages[-1]['content'].count(raw_packet) != 1:
        raise ValueError('Raw evidence must actually reach the model, not just compilation')
    wrapped = '<character_memory trust="untrusted" purpose="historical_reference">\n'
    original = wrapped + escape(compiled['reference_context'], quote=False) + '\n</character_memory>'
    if messages[-1]['content'].count(original) != 1:
        raise ValueError('Observed request must contain the exact compiled reference')
    reduced = copy.deepcopy(messages)
    reduced[-1]['content'] = reduced[-1]['content'].replace(original, '')
    return {'full': copy.deepcopy(messages), 'without_observation_copy': reduced,
            'without_assistant_history': [copy.deepcopy(m) for m in messages if m['role'] != 'assistant'],
            'without_both': [copy.deepcopy(m) for m in reduced if m['role'] != 'assistant']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    root = Path('/home/boot/lhm/multipersonal-runtime/evaluations').resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or not output.name.startswith('r116-'):
        raise ValueError('Requires a new isolated r116 evaluation directory')
    if not 1 <= args.repeats <= 10:
        raise ValueError('Use 1 to 10 repeats')
    from dotenv import load_dotenv

    load_dotenv('/home/boot/lhm/multi-personal-chat/backend/.env', override=False)
    output.mkdir(exist_ok=False)
    calls = 0
    for trace in args.trace:
        if not trace.resolve().is_relative_to(root):
            raise ValueError('Only isolated evaluation traces are accepted')
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            if row.get('phase') != 'question' or not row.get('model_calls'):
                continue
            variants = request_variants(row)
            observed = row['model_calls'][0]['request']
            if observed.get('lora_name'):
                raise ValueError('No LoRA allowed')
            for seed in range(args.repeats):
                modes = list(variants)
                if seed % 2:
                    modes.reverse()
                for mode in modes:
                    body = {key: observed[key] for key in ('temperature', 'max_tokens', 'top_p',
                                                           'repetition_penalty', 'frequency_penalty')}
                    body.update(model='qwen3-8b-instruct-awq', messages=variants[mode], seed=seed,
                                stream=False, chat_template_kwargs={'enable_thinking': observed['enable_thinking']})
                    headers = {'Content-Type': 'application/json'}
                    if os.getenv('VLLM_API_KEY'):
                        headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
                    started = time.monotonic()
                    with urlopen(Request('http://127.0.0.1:8001/v1/chat/completions',
                            data=json.dumps(body).encode(), headers=headers), timeout=120) as response:
                        answer = json.load(response)
                    result = dict(trace=str(trace), case=row['case'], index=row['index'], seed=seed,
                                  mode=mode, request=body, response=answer, seconds=time.monotonic()-started)
                    with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    calls += 1
                    print(json.dumps(dict(source=trace.parent.name, seed=seed, mode=mode,
                        reply=answer['choices'][0]['message']['content']), ensure_ascii=False), flush=True)
    (output / 'manifest.json').write_text(json.dumps(dict(real_model=True, frozen_requests=True,
        application_chain=False, writer_calls=0, generation_calls=calls, system_prompts_unchanged=True,
        source_quotes_preserved=True, lora=False, production_modified=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
