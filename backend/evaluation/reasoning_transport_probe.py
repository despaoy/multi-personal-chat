"""Real client transport smoke probe; not a persona or memory quality test."""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path


async def run(args):
    from inference.vllm_client import VLLMClient

    args.output.mkdir(parents=True, exist_ok=False)
    client = VLLMClient(base_urls=args.base_url, model=args.model)
    messages = [dict(role='user', content='请计算17加26，只给最终结果。')]
    try:
        for budget in (8, 1024):
            for streaming in (False, True):
                started = time.monotonic()
                result = await client.generate(messages, temperature=0, max_tokens=budget,
                                               stream=streaming, enable_thinking=True)
                chunks = [chunk async for chunk in result] if streaming else [result]
                text = ''.join(chunks)
                record = dict(stream=streaming, max_tokens=budget, text=text,
                    visible_chunks=len(chunks), visible_chars=len(text), seconds=time.monotonic()-started,
                    inline_marker_visible='<think>' in text.lower() or '</think>' in text.lower())
                with (args.output / 'transport.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps({key: record[key] for key in ('stream', 'max_tokens', 'visible_chars',
                    'seconds', 'inline_marker_visible')}), flush=True)
    finally:
        await client.close()
    backend = Path(__file__).resolve().parents[1]
    files = [Path(__file__), backend / 'inference/vllm_client.py', backend / 'inference/reasoning_text.py']
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        persona_evaluation=False, full_chain=False), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
