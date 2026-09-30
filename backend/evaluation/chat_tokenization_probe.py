"""Compare actual server tokenization with the saved model's chat template.

Read-only model endpoint calls, no inference or model loading. This certifies
tokenization of supplied inputs, not checkpoint weights or semantic quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation.episode_subject_audit import complete


def token_difference(expected, actual):
    if (not isinstance(expected, list) or not isinstance(actual, list)
            or any(type(value) is not int for value in expected + actual)):
        raise ValueError('Expected integer token lists')
    first = next((i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]), None)
    if first is None and len(expected) != len(actual):
        first = min(len(expected), len(actual))
    return dict(equal=first is None, first_difference=first,
                local_count=len(expected), server_count=len(actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', type=Path, nargs='+', required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for path in args.replays for line in path.read_text(encoding='utf-8').splitlines()]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    seen = set()
    for row in rows:
        request = row['request']
        kwargs = request.get('chat_template_kwargs') or {}
        body = dict(model=request['model'], messages=request['messages'], chat_template_kwargs=kwargs,
                    add_generation_prompt=True, add_special_tokens=False)
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rendered = tokenizer.apply_chat_template(request['messages'], tokenize=False,
                                                  add_generation_prompt=True, **kwargs)
        local = tokenizer.encode(rendered, add_special_tokens=False)
        remote = complete(args.endpoint, body)
        comparison = token_difference(local, remote['tokens'])
        result = dict(case=row['case'], variant=row['variant'], input_sha256=key,
                      comparison=comparison, server_reported_count=remote['count'],
                      prior_generation_count=row['response']['usage']['prompt_tokens'],
                      count_matches_prior=remote['count'] == row['response']['usage']['prompt_tokens'],
                      local_rendered=rendered, local_tokens=local, server_tokens=remote['tokens'])
        results.append(result)
        with (args.output / 'tokenization.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
    files = [Path(__file__), *args.replays, args.model_path / 'tokenizer_config.json', args.model_path / 'tokenizer.json']
    summary = dict(unique_inputs=len(results), model_calls=0, tokenize_calls=len(results),
                   all_token_ids_match=all(r['comparison']['equal'] for r in results),
                   all_prior_counts_match=all(r['count_matches_prior'] for r in results),
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
