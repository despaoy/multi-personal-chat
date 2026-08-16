#!/usr/bin/env python3
"""Generate the 20 overfit responses from a trained PEFT adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    results = []
    for case in cases:
        messages = [{"role": "system", "content": system_prompt}]
        interlocutor = case.get("interlocutor")
        if interlocutor:
            messages[0]["content"] += f"\n\n当前对话者：{interlocutor}。"
        messages.extend(case["messages"])
        template_args = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            inputs = tokenizer.apply_chat_template(messages, enable_thinking=False, **template_args)
        except TypeError:
            inputs = tokenizer.apply_chat_template(messages, **template_args)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, inputs.shape[-1]:], skip_special_tokens=True).strip()
        results.append({**case, "response": response})

    payload = {"schema_version": 1, "status": "pending_human_review", "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
