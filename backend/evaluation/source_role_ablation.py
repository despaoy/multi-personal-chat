"""Diagnose source-packet placement on frozen real model requests.

Not a runtime policy: retrieved speech is not necessarily contiguous history.
The speech-history arm deliberately tests that representation tradeoff. All
variants keep the system, current question and other memory views unchanged.
Packet-only mode preserves warm history; reference-boundary mode moves all
reference blocks together without favoring RAG over user sources.
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

from evaluation.source_packet_placement import separate_reference_packet, separate_source_packet


def packet_request_variants(row: dict) -> dict[str, list[dict]]:
    """Pair the actual request with packet placement, preserving warm history."""
    messages = row['model_calls'][0]['request']['messages']
    source = row['prepared']['compiled']['episodic_reference_context']
    if not source:
        raise ValueError('Source evidence is required for an informative pair')
    return {'full': copy.deepcopy(messages), 'separate_packet': separate_source_packet(messages, source)}


def request_variants(row: dict) -> dict[str, list[dict]]:
    messages = row['model_calls'][0]['request']['messages']
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('Expected a final user query')
    if any(message['role'] != 'system' for message in messages[:-1]):
        raise ValueError('Cold reads only: history must not be reordered or duplicated')
    compiled = row['prepared']['compiled']
    source = compiled['episodic_reference_context']
    raw = json.loads(source)
    if raw.get('speaker_role') != 'user' or raw.get('source_kind') != 'historical_user_utterances':
        raise ValueError('Only captured user sources qualify')
    records = raw['records']
    if not records or any(not isinstance(r.get('text'), str) or not r['text'].strip()
                          or not r.get('source_id') or not r.get('observed_at') for r in records):
        raise ValueError('Complete identified source records required')
    packet = ('<dialogue_evidence trust="untrusted" purpose="historical_utterances">\n'
              + escape(source, quote=False) + '\n</dialogue_evidence>')
    if messages[-1]['content'].count(packet) != 1:
        raise ValueError('The exact source packet must reach the model once')
    final = copy.deepcopy(messages[-1])
    final['content'] = final['content'].replace(packet, '')
    prefix = copy.deepcopy(messages[:-1])
    return {
        'full': copy.deepcopy(messages),
        'separate_packet': [*copy.deepcopy(prefix), {'role': 'user', 'content': packet}, copy.deepcopy(final)],
        'speech_history': [*copy.deepcopy(prefix),
                           *({'role': 'user', 'content': r['text']} for r in
                             sorted(records, key=lambda r: (r['observed_at'], r['source_id']))),
                           copy.deepcopy(final)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--packet-only', action='store_true', help='Preserve warm history; no speech-history arm')
    parser.add_argument('--reference-boundary', action='store_true', help='Also isolate all reference data from the query')
    args = parser.parse_args()
    root = Path('/home/boot/lhm/multipersonal-runtime/evaluations').resolve()
    output = args.output.resolve()
    if (not output.is_relative_to(root) or not output.name.startswith(('r117-', 'r118-'))
            or not args.trace.resolve().is_relative_to(root)):
        raise ValueError('Only isolated r117/r118 output and evaluation inputs accepted')
    if not 1 <= args.repeats <= 10:
        raise ValueError('Use 1 to 10 repeats')
    rows = [json.loads(line) for line in args.trace.read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r.get('phase') == 'question' and r.get('case') in args.case]
    if {r['case'] for r in rows} != set(args.case):
        raise ValueError('Every requested case must be present')
    build_variants = packet_request_variants if args.packet_only else request_variants
    prepared = [(row, build_variants(row)) for row in rows]
    if args.reference_boundary:
        for row, variants in prepared:
            variants['separate_references'] = separate_reference_packet(
                row['model_calls'][0]['request']['messages'],
                row['prepared']['compiled']['episodic_reference_context'])
    from dotenv import load_dotenv

    load_dotenv('/home/boot/lhm/multi-personal-chat/backend/.env', override=False)
    output.mkdir(exist_ok=False)
    calls = 0
    for row, variants in prepared:
        observed = row['model_calls'][0]['request']
        if observed.get('lora_name'):
            raise ValueError('No LoRA allowed')
        for seed in range(args.repeats):
            modes = list(variants)
            if seed % 2:
                modes.reverse()
            for mode in modes:
                body = {k: observed[k] for k in ('temperature', 'max_tokens', 'top_p',
                                                'repetition_penalty', 'frequency_penalty')}
                body.update(model='qwen3-8b-instruct-awq', messages=variants[mode], seed=seed, stream=False,
                            chat_template_kwargs={'enable_thinking': observed['enable_thinking']})
                headers = {'Content-Type': 'application/json'}
                if os.getenv('VLLM_API_KEY'):
                    headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
                started = time.monotonic()
                with urlopen(Request('http://127.0.0.1:8001/v1/chat/completions',
                        data=json.dumps(body).encode(), headers=headers), timeout=120) as response:
                    answer = json.load(response)
                result = dict(case=row['case'], seed=seed, mode=mode, request=body,
                              response=answer, seconds=time.monotonic()-started)
                with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                calls += 1
                print(json.dumps(dict(case=row['case'], mode=mode, seed=seed,
                    reply=answer['choices'][0]['message']['content']), ensure_ascii=False), flush=True)
    (output / 'manifest.json').write_text(json.dumps(dict(real_model=True, frozen_requests=True,
        application_chain=False, writer_calls=0, generation_calls=calls, system_prompts_unchanged=True,
        current_query_unchanged=True, source_bodies_preserved=True, lora=False,
        speech_history_is_diagnostic_not_runtime=not args.packet_only,
        packet_only=args.packet_only, reference_boundary=args.reference_boundary,
        production_modified=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
