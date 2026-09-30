"""Offline specialist NLI feasibility, never runtime truth/admission authority."""
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

LABELS = {'entailment', 'neutral', 'contradiction'}


def validate_cases(cases):
    if not cases or len({row['id'] for row in cases}) != len(cases):
        raise ValueError('Expected unique nonempty independent cases')
    for row in cases:
        if row['label'] not in LABELS or not all(isinstance(row[k], str) and row[k].strip()
                                               for k in ('id', 'group', 'premise', 'hypothesis')):
            raise ValueError('Invalid NLI case')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.suite.read_text(encoding='utf-8'))
    validate_cases(cases)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path,
        local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32).eval()
    names = {int(k): v for k, v in model.config.id2label.items()}
    if set(names.values()) != LABELS:
        raise ValueError('Unknown semantic label mapping')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for row in cases:
        # Gold never enters tokenizer/model; no truncation of critical qualifiers.
        inputs = tokenizer(row['premise'], row['hypothesis'], return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[-1] > model.config.max_position_embeddings:
            raise ValueError('Complete source does not fit; refuse silent clipping')
        with torch.inference_mode():
            if not results:
                model(**inputs)  # one excluded warmup
            start = time.monotonic()
            logits = model(**inputs).logits[0]
            probabilities = logits.softmax(-1).tolist()
            elapsed = time.monotonic() - start
        probabilities = {names[i]: p for i, p in enumerate(probabilities)}
        predicted = max(probabilities, key=probabilities.get)
        record = dict(row, predicted=predicted, correct=predicted == row['label'],
            probabilities=probabilities, seconds=elapsed,
            tokens=inputs['input_ids'].shape[-1],
            false_high_entailment=row['label'] != 'entailment' and probabilities['entailment'] >= .9)
        results.append(record)
        with (args.output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = dict(cases=len(results), correct=sum(r['correct'] for r in results),
        false_high_entailment=sum(r['false_high_entailment'] for r in results),
        gold_counts=dict(Counter(r['label'] for r in results)),
        mean_seconds=statistics.mean(r['seconds'] for r in results), runtime_enabled=False,
        model_calls=len(results)+1, warmup_excluded=True, model_path=str(args.model_path),
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.suite, Path(__file__))})
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
