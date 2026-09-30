"""Frozen-input scope annotation/duplicate-observation diagnostic, not runtime policy."""
import argparse
import copy
import json
from dataclasses import replace
from html import escape
from pathlib import Path

from character.context_builder import compile_reference_context
from character.models import MemoryItem
from evaluation.episode_subject_audit import complete


def variants(row):
    compiled = row['prepared']['compiled']
    items = tuple(MemoryItem(**item) for item in compiled['memory_packets'])
    if not items or not all(item.source_observation for item in items):
        raise ValueError('Only observation-only references supported')
    current, _ = compile_reference_context(items, observation_semantics=True)
    if current != compiled['reference_context']:
        raise ValueError('Reference reconstruction differs')
    source = json.loads(compiled['episodic_reference_context'])
    for item in items:
        if not any(record['source_id'] in item.source_message_ids
                   and record['text'] in item.evidence and record['observed_at'] == item.observed_at
                   for record in source['records']):
            raise ValueError('Source packet must independently contain the same complete observation')
    old, _ = compile_reference_context(tuple(replace(item, source_observation=False) for item in items))
    original = row['model_calls'][0]['request']['messages']
    encoded = escape(current, quote=False)
    if sum(m['content'].count(encoded) for m in original) != 1:
        raise ValueError('Reference must occur exactly once')
    result = {}
    for mode, reference in [('annotated', current), ('legacy', old), ('source_only', '')]:
        messages = copy.deepcopy(original)
        for message in messages:
            message['content'] = message['content'].replace(encoded, escape(reference, quote=False))
        result[mode] = messages
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/boot/lhm/multipersonal-runtime/evaluations').resolve()
    if not args.output.resolve().is_relative_to(root) or not args.output.name.startswith('r123-'):
        raise ValueError('Only isolated r123 output allowed')
    from dotenv import load_dotenv
    load_dotenv('/home/boot/lhm/multi-personal-chat/backend/.env', override=False)
    rows = [json.loads(line) for line in args.trace.read_text(encoding='utf-8').splitlines()]
    row, = [r for r in rows if r['phase'] == 'question' and r['case'] == 'sibling_split']
    modes = variants(row)
    args.output.mkdir(exist_ok=False)
    for seed in range(3):
        for mode, messages in modes.items():
            body = {k: row['model_calls'][0]['request'][k] for k in
                    ('temperature', 'max_tokens', 'top_p', 'repetition_penalty', 'frequency_penalty')}
            body.update(model='qwen3-8b-instruct-awq', messages=messages, seed=seed, stream=False,
                        chat_template_kwargs={'enable_thinking': False})
            response = complete('http://127.0.0.1:8001/v1/chat/completions', body)
            with (args.output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(mode=mode, seed=seed, request=body, response=response), ensure_ascii=False)+'\n')
            print(json.dumps(dict(mode=mode, seed=seed, reply=response['choices'][0]['message']['content']),
                             ensure_ascii=False), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(dict(generation_calls=9, writer_calls=0,
        application_chain=False, system_unchanged=True, source_packet_preserved=True)), encoding='utf-8')


if __name__ == '__main__':
    main()
