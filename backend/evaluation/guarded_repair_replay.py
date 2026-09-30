"""Freeze a failed first answer and test the existing repair branch with a real model.

No retrieval/writer/database/HTTP lifecycle. The recorded actual model input is
the fixed plan, including any prior diagnostic placement. Only the additional
repair generation is fresh; don't count the frozen first answer as a new call.
"""
import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen


async def replay(row, seed):
    from character.models import CompiledCharacterContext, MemoryItem
    from character.output_guard import ReplyGuard
    from inference.generation_request import (
        GenerationPlan,
        GenerationRequest,
        RetrievalResult,
        generate_character_response,
    )

    context = dict(row['prepared']['compiled'])
    context['memory_packets'] = tuple(MemoryItem(**item) for item in context['memory_packets'])
    raw_plan = dict(row['generation']['plan'])
    raw_plan['retrieval'] = RetrievalResult(**raw_plan['retrieval'])
    observed = row['model_calls'][0]
    raw_plan['messages'] = tuple(observed['request']['messages'])
    plan = GenerationPlan(**raw_plan)
    if plan.lora_name:
        raise ValueError('No LoRA allowed')
    request = GenerationRequest(message=row['message'], character_context=CompiledCharacterContext(**context),
        reply_guard=ReplyGuard(**row['prepared']['reply_guard']), retrieval=plan.retrieval,
        history=row['prepared']['history'])
    attempted = 0
    fresh = []

    async def model(**kwargs):
        nonlocal attempted
        attempted += 1
        if attempted == 1:
            if kwargs['messages'] != observed['request']['messages']:
                raise ValueError('Frozen first request changed')
            return observed['reply']
        if attempted > 2:
            raise ValueError('More than one repair requested')
        body = {k: kwargs[k] for k in ('messages', 'temperature', 'max_tokens', 'top_p',
                                      'repetition_penalty', 'frequency_penalty')}
        body.update(model='qwen3-8b-instruct-awq', stream=False, seed=seed,
                    chat_template_kwargs={'enable_thinking': kwargs['enable_thinking']})
        headers = {'Content-Type': 'application/json'}
        if os.getenv('VLLM_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
        at = time.monotonic()
        with urlopen(Request('http://127.0.0.1:8001/v1/chat/completions',
                data=json.dumps(body).encode(), headers=headers), timeout=120) as response:
            answer = json.load(response)
        fresh.append(dict(request=body, response=answer, seconds=time.monotonic()-at))
        return answer['choices'][0]['message']['content']

    with patch('inference.generation_request.build_generation_request', return_value=plan):
        result = await generate_character_response(request, model)
    return dict(case=row['case'], message=row['message'], seed=seed, frozen_first_reply=observed['reply'],
                original_result=row['response']['reply'], result=asdict(result), fresh_calls=fresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/boot/lhm/multipersonal-runtime/evaluations').resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or not output.name.startswith('r119-'):
        raise ValueError('Requires an isolated r119 output directory')
    rows = []
    for path in args.trace:
        if not path.resolve().is_relative_to(root):
            raise ValueError('Only evaluation traces accepted')
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    rows = [row for row in rows if row.get('phase') == 'question'
            and row.get('generation', {}).get('guard_fallback') == 'unsupported_user_fact'
            and row.get('prepared', {}).get('compiled', {}).get('episodic_reference_context')]
    if not rows:
        raise ValueError('No recorded source-bearing guard failures')
    from dotenv import load_dotenv
    load_dotenv('/home/boot/lhm/multi-personal-chat/backend/.env', override=False)
    output.mkdir(exist_ok=False)
    count = 0
    for row in rows:
        for seed in range(3):
            result = asyncio.run(replay(row, seed))
            with (output / 'results.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            count += len(result['fresh_calls'])
            print(json.dumps(dict(case=row['case'], seed=seed, reply=result['result']['reply'],
                fallback=result['result']['guard_fallback'], fresh_calls=len(result['fresh_calls'])),
                ensure_ascii=False), flush=True)
    (output / 'manifest.json').write_text(json.dumps(dict(frozen_first_replies=len(rows),
        trials=len(rows)*3, actual_generation_calls=count, writer_calls=0, application_chain=False,
        actual_shared_guard_pipeline=True, request_builder_frozen=True, production_modified=False,
        lora=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
