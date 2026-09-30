"""Run recorded cloud replies through the real generation/guard pipeline.

No network, writer, database mutation or new generation. Upstream prepared
state and the model-facing plan are frozen. Requiring exactly one model call
detects unexpected early response paths or new retries rather than hiding them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from character.models import CompiledCharacterContext, MemoryItem
from character.output_guard import ReplyGuard
from inference.generation_request import (
    GenerationPlan,
    GenerationRequest,
    RetrievalResult,
    generate_character_response,
)


async def replay(row, cloud):
    compiled = dict(row['prepared']['compiled'])
    compiled['memory_packets'] = tuple(MemoryItem(**m) for m in compiled.get('memory_packets', []))
    retrieval = RetrievalResult(**row['generation']['plan']['retrieval'])
    body = cloud['request']
    plan = GenerationPlan(messages=tuple(body['messages']), retrieval=retrieval,
        generation={key: body[key] for key in ('temperature', 'max_tokens', 'top_p', 'frequency_penalty')},
        prompt_policy_version=row['generation']['plan']['prompt_policy_version'], lora_name=None)
    request = GenerationRequest(message=row['message'], history=row['prepared']['history'],
        character_context=CompiledCharacterContext(**compiled), retrieval=retrieval,
        reply_guard=ReplyGuard(**row['prepared']['reply_guard']))
    calls = []
    reply = cloud['response']['choices'][0]['message']['content']

    async def recorded(**kwargs):
        calls.append(kwargs)
        if len(calls) != 1:
            raise AssertionError('Unexpected retry; no additional model reply is available')
        if kwargs['messages'] != body['messages']:
            raise AssertionError('Frozen input changed')
        return reply

    with patch('inference.generation_request.build_generation_request', return_value=plan):
        result = await generate_character_response(request, recorded)
    if len(calls) != 1:
        raise AssertionError('Unexpected early/deterministic answer; not a model reply replay')
    return dict(case=row['case'], recorded_cloud_reply=reply, reply_preserved=result.reply == reply,
        previous_guard=cloud['guard'], result=asdict(result), new_model_calls=0,
        recorded_model_calls=len(calls), generation_pipeline=True, upstream_frozen=True,
        application_http=False)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-results', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    with args.output.open('x', encoding='utf-8') as out:
        for path in args.cloud_results:
            for line in path.read_text(encoding='utf-8').splitlines():
                cloud = json.loads(line)
                rows = [json.loads(line) for line in Path(cloud['source']).read_text(encoding='utf-8').splitlines()]
                row, = [r for r in rows if r.get('case') == cloud['case'] and r.get('phase') == 'question']
                result = await replay(row, cloud)
                out.write(json.dumps(result, ensure_ascii=False)+'\n')
                results.append(result)
    print(json.dumps(dict(cases=len(results), preserved=sum(r['reply_preserved'] for r in results),
        new_model_calls=0, retries=sum(r['result']['guard_retried'] for r in results)), ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
