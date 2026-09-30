"""Bounded CPU unquantized/served AWQ diagnosis using existing local weights.

Never changes or starts a server; not an isolated quantization causality claim
because the checkpoints and inference backends may differ.
"""
import argparse
import json
import os
import time
from pathlib import Path

from evaluation.episode_subject_audit import complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/boot/lhm').resolve()
    if not args.model_path.resolve().is_relative_to(root) or not args.output.resolve().is_relative_to(root):
        raise ValueError('Only scoped local model/output paths permitted')
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Float32 permits normal CPU kernels without depending on BF16 hardware.
    # 8B weights ~33GB plus loading/activation reserve; refuse on a busy host.
    if psutil.virtual_memory().available < 48 * 1024**3:
        raise RuntimeError('Insufficient available RAM for isolated CPU comparison')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    rows = [json.loads(line) for line in args.replays.read_text(encoding='utf-8').splitlines()]
    cases = [r for r in rows if r['variant'] == 'core_plain' and r['seed'] == 0]
    if len(cases) != 8 or len({r['case'] for r in cases}) != 8:
        raise ValueError('Expected exactly eight independent core controls')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    started = time.monotonic()
    print('CPU_LOADING existing local base, float32, two threads, no GPU', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, device_map='cpu',
        low_cpu_mem_usage=True, attn_implementation='sdpa').eval()
    loading_seconds = time.monotonic() - started
    print('CPU_LOADED', flush=True)
    for row in cases:
        messages = row['request']['messages']
        encoded = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors='pt')
        if encoded.shape[-1] > 200:
            raise ValueError('CPU experiment must use short core inputs only')
        start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(encoded, attention_mask=torch.ones_like(encoded),
                do_sample=False, max_new_tokens=96, repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id)
        completion = generated[0, encoded.shape[-1]:].tolist()
        cpu = dict(text=tokenizer.decode(completion, skip_special_tokens=True),
                   input_tokens=encoded.shape[-1], output_tokens=len(completion),
                   output_token_ids=completion, seconds=time.monotonic()-start)
        request = dict(model=row['request']['model'], messages=messages, temperature=0,
            top_p=1.0, max_tokens=96, repetition_penalty=1.0, frequency_penalty=0.0,
            chat_template_kwargs={'enable_thinking': False}, seed=0)
        start = time.monotonic()
        response = complete(args.endpoint, request)
        record = dict(case=row['case'], messages=messages, cpu=cpu,
            awq_response=response, awq_seconds=time.monotonic()-start, awq_request=request,
            cpu_input_token_ids=encoded[0].tolist())
        with (args.output / 'comparisons.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(dict(case=row['case'], cpu_seconds=cpu['seconds'])), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        completed_cases=len(cases), model_path=str(args.model_path), cpu_threads=2,
        cpu_dtype='float32', loading_seconds=loading_seconds, gpu_used_by_local_model=False,
        generation='greedy', max_new_tokens=96, thinking=False,
        new_downloads=False, production_modified=False,
        limitation='Checkpoint revisions and inference backends are not held identical; not proof of quantization causality.'
    ), indent=2), encoding='utf-8')
    print('CPU_COMPARISON_DONE', flush=True)


if __name__ == '__main__':
    main()
