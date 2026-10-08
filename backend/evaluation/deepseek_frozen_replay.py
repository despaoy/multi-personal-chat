"""Compare DeepSeek on recorded public/synthetic inputs without changing production.

Credentials are read interactively, never saved. This freezes upstream memory,
retrieval and prompts; guard inspection is diagnostic, not a full HTTP replay.
"""
from __future__ import annotations

import argparse
import getpass
import json
import time
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from character.output_guard import (
    UNSUPPORTED_USER_FACT,
    ReplyGuard,
    ground_reply_guard,
    retryable_violations,
    validate_reply,
)


def payload(request, model):
    # Do not forward local vLLM/LoRA/chat-template controls to a cloud provider.
    return dict(model=model, messages=request['messages'], stream=False,
        max_tokens=request['max_tokens'], temperature=request['temperature'],
        top_p=request['top_p'], frequency_penalty=request['frequency_penalty'],
        thinking={'type': 'disabled'})


def inspect_guard(row, reply):
    guard = ReplyGuard(**row['prepared']['reply_guard'])
    retrieval = row['generation']['plan']['retrieval']
    if retrieval.get('status') == 'ok' and retrieval.get('evidence'):
        guard = ground_reply_guard(guard, retrieval['evidence'])
    violations = validate_reply(reply, guard)
    blocking = retryable_violations(reply, guard, violations, strict=False)
    result = dict(violations=violations, blocking=blocking, action='pass', final_if_no_retry=reply)
    if blocking:
        result.update(action='would_retry', final_if_no_retry=None)
    result['forced_preference_detector'] = UNSUPPORTED_USER_FACT in validate_reply(
        reply, replace(guard, forbid_unsupported_user_fact=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-v4-pro')
    args = parser.parse_args()
    rows = [dict(source=str(path), **json.loads(line)) for path in args.trace
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if r.get('phase') == 'question' and r.get('model_calls')]
    if not 1 <= len(rows) <= 12:
        raise ValueError('This initial comparison accepts 1..12 recorded questions')
    args.output.mkdir(exist_ok=False)
    api_key = getpass.getpass('DeepSeek API key (not saved): ')

    def call(path, body=None):
        request = Request('https://api.deepseek.com'+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Authorization': 'Bearer '+api_key, 'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=150) as response:
                return json.load(response)
        except HTTPError as exc:
            # Avoid logging request headers or provider error bodies.
            raise RuntimeError(f'DeepSeek HTTP {exc.code}') from None

    models = [entry['id'] for entry in call('/models')['data']]
    print(json.dumps(dict(available_models=models), ensure_ascii=False), flush=True)
    if args.model not in models:
        raise ValueError('Requested model is not available; choose an advertised ID')
    manifest = dict(model=args.model, planned_calls=len(rows), completed_calls=0,
        application_http=False, upstream_frozen=True, writer_calls=0, production_modified=False,
        lora=False, thinking=False, retries_executed=False, messages_unchanged=True,
        public_synthetic_inputs=True, parameters_removed=['lora_name', 'repetition_penalty', 'enable_thinking'])
    try:
        for row in rows:
            body = payload(row['model_calls'][0]['request'], args.model)
            start = time.monotonic()
            response = call('/chat/completions', body)
            elapsed = time.monotonic()-start
            reply = response['choices'][0]['message']['content']
            item = dict(case=row['case'], source=row['source'], message=row['message'],
                request=body, response=response, seconds=elapsed,
                baseline_raw=row['model_calls'][0]['reply'], baseline_final=row['response']['reply'],
                guard=inspect_guard(row, reply))
            with (args.output/'results.jsonl').open('a', encoding='utf-8') as out:
                out.write(json.dumps(item, ensure_ascii=False)+'\n')
            manifest['completed_calls'] += 1
            print(json.dumps(dict(case=row['case'], reply=reply, seconds=elapsed,
                guard=item['guard'], finish=response['choices'][0]['finish_reason'],
                usage=response.get('usage')), ensure_ascii=False), flush=True)
    finally:
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
