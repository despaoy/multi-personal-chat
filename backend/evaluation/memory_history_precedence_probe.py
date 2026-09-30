"""Real-model replay of the saved-memory fast-path/history precedence contract.

Synthetic compiled packets, real shared generation/guard, no repository or HTTP
application E2E. No persistence, prompts or model settings changed by the probe.
"""
import argparse
import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from character.memory_extractor import extract_memories
from character.models import CompiledCharacterContext, MemoryItem
from evaluation.episode_subject_audit import complete
from inference.generation_request import GenerationRequest, generate_character_response

CASES = (
    ('residence_update', '我住在银川。', '我现在住在西宁。', '我住哪里？'),
    ('major_update', '我的专业是地质学。', '我的专业改成了天文学。', '我的专业是什么？'),
    ('name_update', '我叫林溪。', '我现在改名叫陈霁了。', '我的名字是什么？'),
    ('work_update', '我在观测站工作。', '我已经离开观测站，现在在博物馆工作。', '我的工作地点是什么？'),
    ('owner_correction', '我住在银川。', '上次那个地址不是我的，是我朋友的。', '我住哪里？'),
    ('value_withdrawal', '我住在银川。', '那个地址已经不对了，新的我以后再告诉你。', '我住哪里？'),
    ('storage_inventory', '我住在银川。', '旧住址不对了。', '你保存了我的现居地吗？'),
)


def request(case):
    name, old, update, query = case
    fact, = extract_memories(old)
    item = MemoryItem('saved', 'user_fact', fact.content, memory_key=fact.memory_key,
        evidence=(old,), source_message_ids=('old-synthetic-source',))
    ctx = CompiledCharacterContext('', '', fact.content, ('saved',),
        memory_status='available', memory_packets=(item,), memory_field_presence=(('residence', True),))
    history = ({'role': 'user', 'content': update},)
    return GenerationRequest(message=query, character_context=ctx, history=history, max_tokens=256)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [request(case) for case in CASES]
    # Controls protect the cheap, correct paths without changing the facts.
    cases.extend([replace(cases[0], history=()),
                  replace(cases[0], history=({'role': 'assistant', 'content': '你住在杭州。'},))])
    names = [case[0] for case in CASES] + ['no_history_control', 'assistant_only_control']
    for name, req in zip(names, cases):
        calls = []

        async def generate(_calls=calls, **kwargs):
            body = dict(model=args.model, messages=kwargs['messages'], temperature=kwargs['temperature'],
                top_p=kwargs['top_p'], max_tokens=kwargs['max_tokens'],
                repetition_penalty=kwargs['repetition_penalty'], frequency_penalty=kwargs['frequency_penalty'],
                chat_template_kwargs={'enable_thinking': kwargs['enable_thinking']}, seed=0)
            response = await asyncio.to_thread(complete, args.endpoint, body)
            _calls.append(dict(request=body, response=response))
            return response['choices'][0]['message']['content']

        start = time.monotonic()
        result = await generate_character_response(req, generate)
        with (args.output / 'results.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(case=name, request=asdict(req), result=asdict(result),
                model_calls=calls, seconds=time.monotonic()-start), ensure_ascii=False, default=str) + '\n')
        print(json.dumps(dict(case=name, calls=len(calls), mode=result.response_mode)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
