#!/usr/bin/env python3
"""Generate post-training Kisaki V4 chat-smoke responses from one adapter."""

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
    parser.add_argument("--variant", required=True)
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
    payload = json.loads(args.cases.read_text(encoding="utf-8"))

    def generate(messages, interlocutor):
        conversation = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n当前对话者：{interlocutor}。",
            },
            *messages,
        ]
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            inputs = tokenizer.apply_chat_template(conversation, enable_thinking=False, **kwargs)
        except TypeError:
            inputs = tokenizer.apply_chat_template(conversation, **kwargs)
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            output = model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True).strip()

    natural = []
    for case in payload["natural_chat"]:
        natural.append({**case, "response": generate(case["messages"], case["interlocutor"])})

    continuity = payload["continuity"]
    history = []
    continuity_results = []
    for turn, user_text in enumerate(continuity["user_turns"], 1):
        history.append({"role": "user", "content": user_text})
        response = generate(history, continuity["interlocutor"])
        continuity_results.append({"turn": turn, "user": user_text, "response": response})
        history.append({"role": "assistant", "content": response})

    contextual = []
    for case in payload["contextual_story"]:
        contextual.append({**case, "response": generate(case["messages"], case["interlocutor"])})

    result = {
        "schema_version": 1,
        "status": "pending_human_review",
        "variant": args.variant,
        "natural_chat": natural,
        "continuity": {**continuity, "turns": continuity_results},
        "contextual_story": contextual,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
