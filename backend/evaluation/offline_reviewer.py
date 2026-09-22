"""Sequential offline evaluator, never a production inference backend.

Uses only existing local safetensors and never executes model-repository code.
Generation runs synchronously in the CLI's event loop: this intentionally avoids
orphaned CUDA threads after asyncio cancellation. Its token/time limits are
checked by Transformers between decoding steps, not a hard process deadline.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def validate_generation_budget(input_tokens: int, max_input_tokens: int) -> None:
    if input_tokens <= 0 or input_tokens > max_input_tokens:
        raise ValueError("complete prompt exceeds offline input budget; no truncation permitted")


def final_answer_text(decoded: str, *, thinking: bool) -> str:
    if not thinking:
        return decoded.strip()
    # Qwen emits an explicit reasoning terminator. Do not pass private scratch
    # text into JSON parsers, logs, a second reviewer call or dialogue history.
    if decoded.count("</think>") != 1:
        raise ValueError("thinking generation lacks one unambiguous final-answer boundary")
    final = decoded.split("</think>", 1)[1].strip()
    if not final or "<think>" in final:
        raise ValueError("thinking generation has no final answer")
    return final


class OfflineTransformersReviewer:
    def __init__(
        self,
        model_path: str | Path,
        *,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 1024,
        max_seconds: float = 90.0,
        decoding: str = "greedy",
        enable_thinking: bool = False,
        seed: int = 42,
    ) -> None:
        path = Path(model_path).resolve(strict=True)
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("an existing local model snapshot is required")
        if not 128 <= max_input_tokens <= 8192 or not 32 <= max_new_tokens <= 2048:
            raise ValueError("offline token limits out of bounds")
        if not math.isfinite(max_seconds) or not 1 <= max_seconds <= 120:
            raise ValueError("offline decoding limit must be between 1 and 120 seconds")
        if decoding not in {"greedy", "sampled"} or (enable_thinking and decoding != "sampled"):
            raise ValueError("thinking ablation requires sampled decoding; mode must be greedy or sampled")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("this bounded pilot requires an available CUDA device")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            device_map={"": 0},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.max_seconds = max_seconds
        self.decoding = decoding
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.calls: list[dict] = []
        quantization = getattr(self.model.config, "quantization_config", None)
        if hasattr(quantization, "to_dict"):
            quantization = quantization.to_dict()
        packages = {}
        for package in ("torch", "transformers", "bitsandbytes", "accelerate"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        self.metadata = {
            "backend": "offline_transformers",
            "model_snapshot": path.name,
            "model_family": path.parent.parent.name,
            "quantization": json.loads(json.dumps(quantization, default=str)),
            "packages": packages,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "max_seconds": max_seconds,
            "enable_thinking": enable_thinking,
            "decoding": decoding,
            "seed": seed if decoding == "sampled" else None,
            "sampling_parameters": (
                {
                    "temperature": 0.6 if enable_thinking else 0.7,
                    "top_p": 0.95 if enable_thinking else 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                }
                if decoding == "sampled"
                else None
            ),
            "seed_strategy": "sha256(initial_seed,formatted_prompt); forked RNG; order-independent",
            "production_equivalence": False,
        }

    async def __call__(self, messages):
        import torch

        text = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
        )
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
        input_tokens = encoded["input_ids"].shape[-1]
        validate_generation_budget(input_tokens, self.max_input_tokens)
        encoded = encoded.to(self.model.device)
        started = time.perf_counter()
        sampled = self.decoding == "sampled"
        generation_options = self.metadata["sampling_parameters"] or {}
        rng = torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))) if sampled else nullcontext()
        with torch.inference_mode(), rng:
            if sampled:
                prompt_seed = int.from_bytes(hashlib.sha256(f"{self.seed}\0{text}".encode()).digest()[:6], "big")
                torch.manual_seed(prompt_seed)
            generated = self.model.generate(
                **encoded,
                do_sample=sampled,
                **generation_options,
                max_new_tokens=self.max_new_tokens,
                max_time=self.max_seconds,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )
        new_tokens = generated[0, input_tokens:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        complete = bool(new_tokens) and new_tokens[-1] in eos_ids
        self.calls.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": len(new_tokens),
                "latency_seconds": time.perf_counter() - started,
                "complete": complete,
            }
        )
        if not complete:
            raise TimeoutError("offline generation reached a token or decoding-time limit")
        return final_answer_text(
            self.tokenizer.decode(new_tokens, skip_special_tokens=True), thinking=self.enable_thinking
        )
