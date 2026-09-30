"""Single-device LoRA PPO with a frozen SFT reference and an external reward judge.

Full-distribution sampling (temperature=1, top_p=1, top_k=0) keeps stored action
probabilities identical to the behavior policy used by clipped PPO updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import torch

from training.persona_alignment import read_json, read_jsonl, write_json, write_jsonl
from training.persona_judge import RUBRIC_SHA256, judge_candidate
from training.persona_sampling import (
    digest,
    sample_candidates,
    sampling_contract,
    scene_messages,
    validate_model,
    validate_scenes,
)
from training.ppo_math import action_statistics, clipped_loss, generalized_advantages, sampled_kl
from training.preference_reference import assert_initial_reference, assert_reference_unchanged, reference_fingerprint
from training.preference_validation import validate_adapter_path


@dataclass
class PPOConfig:
    base_model_path: str
    adapter_path: str
    output_dir: str
    base_revision: str = "local-unversioned"
    updates: int = 10
    rollout_batch_size: int = 4
    ppo_epochs: int = 2
    learning_rate: float = 5e-7
    critic_learning_rate: float = 1e-4
    max_prompt_length: int = 1536
    max_new_tokens: int = 128
    max_length: int = 2048
    max_turns: int = 5
    kl_coef: float = 0.05
    max_reference_kl: float = 0.5
    target_policy_kl: float = 0.02
    clip_range: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    load_in_4bit: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42
    eval_every: int = 5
    allow_uncalibrated_pilot: bool = False
    min_calibration_pairs: int = 30
    min_calibration_groups: int = 10
    min_calibration_agreement: float = 0.7
    min_calibration_coverage: float = 0.8

    def validate(self):
        for key in (
            "updates",
            "rollout_batch_size",
            "ppo_epochs",
            "max_prompt_length",
            "max_new_tokens",
            "max_length",
            "max_turns",
            "eval_every",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.max_prompt_length + self.max_new_tokens > self.max_length:
            raise ValueError("prompt + generation budget exceeds max_length")
        for key in (
            "learning_rate",
            "critic_learning_rate",
            "max_grad_norm",
            "target_policy_kl",
            "max_reference_kl",
            "clip_range",
            "value_clip",
            "value_coef",
        ):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("kl_coef", "entropy_coef"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        for key in ("gamma", "gae_lambda", "min_calibration_agreement", "min_calibration_coverage"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        if self.clip_range >= 1 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("clip_range must be <1 and seed must be a nonnegative integer")
        for key in ("min_calibration_pairs", "min_calibration_groups"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("calibration count thresholds must be positive integers")
        validate_adapter_path(self.adapter_path)
        if not self.adapter_path:
            raise ValueError("PPO requires an SFT adapter as policy initialization and reference")
        if not self.base_model_path or not self.base_revision:
            raise ValueError("base model path and revision are required")


class JudgeReward:
    def __init__(self, model, call=None):
        validate_model(model)
        self.model, self.call = model, call
        judge_id = f"{model['model']}@{model['revision']}/{digest(model)[:12]}/{RUBRIC_SHA256[:12]}"
        self.contract = {"kind": "ai_judge", "judge_id": judge_id, "model": model, "rubric_sha256": RUBRIC_SHA256}

    def score(self, record):
        return judge_candidate(record, self.model, **({"call": self.call} if self.call else {}))


def verify_calibration(config, reward_contract, calibration):
    if calibration is None:
        if not config.allow_uncalibrated_pilot:
            raise ValueError("provide a bound calibration report or explicitly enable an uncalibrated pilot")
        return "uncalibrated_pilot"
    report = calibration.get("report", calibration)
    if report.get("status") != "measured_not_approved_for_ppo" or report.get("judge_id") != reward_contract.get(
        "judge_id"
    ):
        raise ValueError("calibration must match the exact judge/version/rubric")
    for field, threshold in (
        ("pair_count", config.min_calibration_pairs),
        ("source_group_count", config.min_calibration_groups),
        ("agreement", config.min_calibration_agreement),
        ("coverage", config.min_calibration_coverage),
    ):
        value = report.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < threshold:
            raise ValueError(f"calibration {field} does not meet the configured pilot threshold")
    return "calibration_thresholds_met_not_release_approval"


def adapter_files_hash(path):
    root, checksum = Path(path), hashlib.sha256()
    files = [root / "adapter_config.json", *sorted(root.glob("adapter_model.*"))]
    for file in files:
        checksum.update(file.name.encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
    return checksum.hexdigest()


def base_model_identity(config):
    """Bind local weights/tokenizer bytes or an immutable Hub commit to a run."""
    root = Path(config.base_model_path)
    if not root.is_dir():
        if not re.fullmatch(r"[0-9a-f]{40}", config.base_revision):
            raise ValueError("remote base model requires an immutable 40-character commit revision")
        return {"model": config.base_model_path, "commit": config.base_revision}
    checksum = hashlib.sha256()
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".tiktoken"}
    )
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError("local base model contains no weight files")
    for file in files:
        checksum.update(file.relative_to(root).as_posix().encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
    return {"sha256": checksum.hexdigest(), "file_count": len(files)}


class PersonaPPO:
    def __init__(self, config: PPOConfig, reward):
        self.config, self.reward = config, reward
        self.model = self.tokenizer = self.critic = self.optimizer = None
        self.reference_sha = ""
        self.update = 0
        self.history = []

    def _load(self, policy_path=None):
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

        config = self.config
        set_seed(config.seed)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if config.load_in_4bit and device == "cpu":
            raise ValueError("4-bit PPO requires a compatible CUDA device; use nonquantized CPU for tiny tests")
        dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32
        quantization = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            if config.load_in_4bit
            else None
        )
        revision = {} if Path(config.base_model_path).is_dir() else {"revision": config.base_revision}
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, **revision)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("PPO tokenizer requires an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=dtype,
            quantization_config=quantization,
            device_map={"": device},
            **revision,
        )
        if getattr(base, "is_loaded_in_4bit", False) or getattr(base, "is_loaded_in_8bit", False):
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=config.gradient_checkpointing)
        self.model = PeftModel.from_pretrained(base, policy_path or config.adapter_path, is_trainable=True)
        self.model.load_adapter(config.adapter_path, adapter_name="reference", is_trainable=False)
        self.model.set_adapter("default")
        self.reference_sha = reference_fingerprint(self.model)
        if policy_path is None:
            assert_initial_reference(self.model)
        # Eval rollouts and differentiable updates must use the same distribution.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.critic = torch.nn.Linear(self.model.config.hidden_size, 1).to(device=device, dtype=torch.float32)
        torch.nn.init.zeros_(self.critic.weight)
        torch.nn.init.zeros_(self.critic.bias)
        policy_parameters = [value for value in self.model.parameters() if value.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": policy_parameters, "lr": config.learning_rate},
                {"params": list(self.critic.parameters()), "lr": config.critic_learning_rate},
            ],
            weight_decay=0,
        )

    def _prompt(self, messages):
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if not 0 < ids.shape[1] <= self.config.max_prompt_length:
            raise ValueError("PPO prompt exceeds budget; history/evidence truncation is forbidden")
        return ids.to(next(self.model.parameters()).device)

    def _generate(self, messages, *, seed, deterministic=False):
        from transformers import GenerationConfig

        prompt = self._prompt(messages)
        torch.manual_seed(seed)
        self.model.eval()
        generation = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=not deterministic,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        with torch.no_grad():
            tokens = self.model.generate(prompt, attention_mask=torch.ones_like(prompt), generation_config=generation)
        actions = tokens[0, prompt.shape[1] :]
        eos = (actions == self.tokenizer.eos_token_id).nonzero()
        if eos.numel():
            actions = actions[: int(eos[0].item()) + 1]
        if not actions.numel():
            raise RuntimeError("PPO generation returned no actions")
        tokens = torch.cat((prompt, actions.unsqueeze(0)), dim=1)
        response = self.tokenizer.decode(actions, skip_special_tokens=True)
        return tokens, prompt.shape[1], response, bool(actions[-1].item() == self.tokenizer.eos_token_id)

    def _collect_episode(self, original_scene, episode_index):
        config, segments, trace = self.config, [], []
        scene = {**original_scene, "history": [dict(message) for message in original_scene.get("history", [])]}
        turns = [original_scene["user_message"], *original_scene.get("continuations", [])]
        for turn, query in enumerate(turns):
            scene = {**scene, "user_message": query}
            messages = scene_messages(scene)
            seed = config.seed + episode_index * config.max_turns + turn
            self.model.set_adapter("default")
            tokens, prompt_length, response, ended = self._generate(messages, seed=seed)
            with torch.no_grad():
                old_log_probs, old_values, _ = action_statistics(self.model, self.critic, tokens, prompt_length)
                self.model.set_adapter("reference")
                try:
                    ref_log_probs, _, _ = action_statistics(self.model, self.critic, tokens, prompt_length)
                finally:
                    self.model.set_adapter("default")
            model_identity = {
                "name": "ppo_policy",
                "model": config.base_model_path,
                "revision": f"{self.run_signature}/update-{self.update}",
                "base_url": "http://local-training.invalid",
            }
            generation = {"temperature": 1.0, "top_p": 1.0, "max_tokens": config.max_new_tokens, "seed": seed}
            run_id = digest([self.run_signature, self.update, episode_index, turn])
            record = {
                "run_id": run_id,
                "scene": scene,
                "messages": messages,
                "model": model_identity,
                "generation": generation,
                "sample_index": 0,
                "status": "generated",
                "mock": False,
                "candidate_id": digest([run_id, scene["id"], model_identity["name"], 0]),
                "context_sha256": digest({"scene": scene, "messages": messages}),
                "request_sha256": digest({"model": model_identity, "messages": messages, "generation": generation}),
                "response": response,
                "response_sha256": digest(response),
            }
            if not ended or not response.strip():
                scalar, judgment = (
                    -1.0,
                    {"status": "rule_penalty", "reason": "length_limit" if not ended else "empty_response"},
                )
            else:
                judgment = self.reward.score(record)
                if judgment.get("judge_id") != self.reward.contract.get("judge_id"):
                    raise ValueError("PPO reward judge identity mismatch")
                for field in ("candidate_id", "context_sha256", "response_sha256"):
                    if judgment.get(field) != record[field]:
                        raise ValueError("PPO reward is bound to another response/context")
                if judgment.get("status") == "abstained":
                    return [], {
                        "scene_id": original_scene["id"],
                        "status": "abstained",
                        "turns": [*trace, {"candidate": record, "judgment": judgment}],
                    }
                if judgment.get("status") != "scored":
                    raise ValueError("PPO requires a scored reward or explicit abstention")
                scalar = judgment["aggregation"]["reward"]
                if type(scalar) not in (int, float) or not math.isfinite(scalar) or not -1 <= scalar <= 1:
                    raise ValueError("PPO reward must be finite and in [-1,1]")
            kl = sampled_kl(old_log_probs, ref_log_probs)
            rewards = -config.kl_coef * kl
            rewards[-1] += scalar
            segments.append(
                {
                    "tokens": tokens.detach(),
                    "prompt_length": prompt_length,
                    "old_log_probs": old_log_probs.detach(),
                    "old_values": old_values.detach(),
                    "rewards": rewards.detach(),
                    "reference_kl": kl.mean().item(),
                    "score": scalar,
                }
            )
            trace.append({"candidate": record, "judgment": judgment, "reward": scalar, "ended_with_eos": ended})
            if not ended or not response.strip():
                break
            # Store exactly the input that preceded the generated assistant turn.
            # Current query is plain; evidence is reattached to the new current turn.
            scene = {
                **scene,
                "history": [
                    *scene.get("history", []),
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
            }
        advantages, returns = generalized_advantages(
            torch.cat([item["rewards"] for item in segments]),
            torch.cat([item["old_values"] for item in segments]),
            gamma=config.gamma,
            lam=config.gae_lambda,
        )
        offset = 0
        for item in segments:
            length = item["old_log_probs"].numel()
            item.update(advantages=advantages[offset : offset + length], returns=returns[offset : offset + length])
            offset += length
        return segments, {"scene_id": original_scene["id"], "status": "collected", "turns": trace}

    def _optimize(self, segments):
        config = self.config
        advantages = torch.cat([item["advantages"] for item in segments])
        mean, std = advantages.mean(), advantages.std(unbiased=False).clamp_min(1e-8)
        for item in segments:
            item["advantages"] = (item["advantages"] - mean) / std
        total_tokens, metrics, epochs = advantages.numel(), {}, 0
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        for _ in range(config.ppo_epochs):
            self.model.set_adapter("default")
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            epoch_metrics = {}
            for item in segments:
                log_probs, values, entropy = action_statistics(
                    self.model, self.critic, item["tokens"], item["prompt_length"]
                )
                loss, batch_metrics = clipped_loss(
                    log_probs,
                    values,
                    entropy,
                    item["old_log_probs"],
                    item["old_values"],
                    item["advantages"],
                    item["returns"],
                    clip_range=config.clip_range,
                    value_clip=config.value_clip,
                    value_coef=config.value_coef,
                    entropy_coef=config.entropy_coef,
                )
                weight = log_probs.numel() / total_tokens
                (loss * weight).backward()
                for name, value in batch_metrics.items():
                    epoch_metrics[name] = epoch_metrics.get(name, 0) + weight * value
            if epoch_metrics["approx_policy_kl"] > config.target_policy_kl:
                self.optimizer.zero_grad(set_to_none=True)
                metrics["early_stop_policy_kl"] = epoch_metrics["approx_policy_kl"]
                break
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            epochs += 1
            metrics = {**epoch_metrics, "grad_norm": float(grad_norm)}
        assert_reference_unchanged(self.model, self.reference_sha)
        return {**metrics, "ppo_epochs_completed": epochs, "assistant_tokens": total_tokens}

    def _checkpoint(self, root):
        target = root / f"checkpoint-{self.update:06d}"
        pending = root / f"checkpoint-{self.update:06d}.pending"
        if target.exists():
            raise ValueError("checkpoint already exists; refusing overwrite")
        pending.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(pending / "policy", selected_adapters=["default"])
        self.tokenizer.save_pretrained(pending / "policy")
        torch.save(
            {
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "update": self.update,
                "history": self.history,
            },
            pending / "state.pt",
        )
        state_hash = hashlib.sha256((pending / "state.pt").read_bytes()).hexdigest()
        write_json(
            pending / "manifest.json",
            {
                "run_signature": self.run_signature,
                "update": self.update,
                "reference_sha256": self.reference_sha,
                "policy_sha256": adapter_files_hash(pending / "policy"),
                "state_sha256": state_hash,
            },
        )
        pending.rename(target)
        return target

    def _restore(self, checkpoint):
        manifest = read_json(checkpoint / "manifest.json")
        if manifest["run_signature"] != self.run_signature:
            raise ValueError("resume configuration/data/reward/source-adapter mismatch")
        if manifest["reference_sha256"] != self.reference_sha or manifest["policy_sha256"] != adapter_files_hash(
            checkpoint / "policy"
        ):
            raise ValueError("checkpoint policy/reference integrity failure")
        if manifest["state_sha256"] != hashlib.sha256((checkpoint / "state.pt").read_bytes()).hexdigest():
            raise ValueError("checkpoint optimizer/critic integrity failure")
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.update, self.history = state["update"], state["history"]
        if self.update != manifest["update"] or self.update > self.config.updates:
            raise ValueError("invalid resume update boundary")

    def _evaluate(self, scenes, directory):
        generation = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.config.max_new_tokens,
            "seed": self.config.seed,
        }
        models = [
            {
                "name": name,
                "model": self.config.base_model_path,
                "revision": revision,
                "base_url": "http://local-training.invalid",
            }
            for name, revision in (
                ("sft_reference", self.reference_sha),
                ("ppo_policy", f"{self.run_signature}/update-{self.update}"),
            )
        ]

        def generate(model, messages, params):
            self.model.set_adapter("reference" if model["name"] == "sft_reference" else "default")
            try:
                _, _, response, ended = self._generate(messages, seed=params["seed"], deterministic=True)
                if not ended:
                    raise ValueError("incomplete evaluation response")
                return response
            finally:
                self.model.set_adapter("default")

        records = sample_candidates(scenes, models, generation, samples_per_model=1, call=generate)
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "sampling_manifest.json", sampling_contract(scenes, models, generation, 1))
        write_jsonl(directory / "candidates.jsonl", records)
        write_json(
            directory / "summary.json",
            {
                "total": len(records),
                "errors": sum(r["status"] != "generated" for r in records),
                "note": "Fixed-history validation responses, awaiting blinded human review.",
            },
        )
        assert_reference_unchanged(self.model, self.reference_sha)

    def preflight(self, scenes, *, validation=None, calibration=None):
        """Validate immutable inputs without loading weights or calling the judge."""
        config = self.config
        config.validate()
        validate_scenes([*scenes, *(validation or [])])
        if not scenes or any(scene["split"] != "train" for scene in scenes):
            raise ValueError("PPO optimization only accepts reviewed train scenes")
        if validation is not None and (not validation or any(scene["split"] != "validation" for scene in validation)):
            raise ValueError("PPO training-time evaluation only accepts validation scenes")
        for scene in [*scenes, *(validation or [])]:
            followups = scene.get("continuations", [])
            if not isinstance(followups, list) or any(not isinstance(q, str) or not q.strip() for q in followups):
                raise ValueError("continuations must be a list of scripted user messages")
            if len(followups) + 1 > config.max_turns:
                raise ValueError("scene exceeds configured max_turns")
        reward_status = verify_calibration(config, self.reward.contract, calibration)
        base_identity = base_model_identity(config)
        immutable = {k: v for k, v in asdict(config).items() if k not in {"output_dir", "updates"}}
        self.run_signature = digest(
            {
                "config": immutable,
                "train": scenes,
                "validation": validation,
                "reward": self.reward.contract,
                "calibration": calibration,
                "source_adapter_sha256": adapter_files_hash(config.adapter_path),
                "base_model_identity": base_identity,
            }
        )
        from transformers import AutoTokenizer

        revision = {} if Path(config.base_model_path).is_dir() else {"revision": config.base_revision}
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, **revision)
        if tokenizer.eos_token_id is None:
            raise ValueError("PPO tokenizer requires an EOS token")
        lengths = [
            len(tokenizer.apply_chat_template(scene_messages(scene), add_generation_prompt=True))
            for scene in [*scenes, *(validation or [])]
        ]
        if any(length > config.max_prompt_length for length in lengths):
            raise ValueError("PPO prompt exceeds budget; history/evidence truncation is forbidden")
        if config.load_in_4bit and not torch.cuda.is_available():
            raise ValueError("4-bit PPO requires a compatible CUDA device")
        return {
            "run_signature": self.run_signature,
            "base_model_identity": base_identity,
            "reward_status": reward_status,
            "max_initial_prompt_tokens": max(lengths),
            "train_scenes": len(scenes),
            "validation_scenes": len(validation or []),
            "note": "Initial prompts checked; generated multi-turn histories are checked during rollout.",
        }

    def train(self, scenes, *, validation=None, calibration=None, resume=None, stop_after=None):
        config = self.config
        if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
            raise ValueError("stop_after must be a positive update boundary")
        checks = self.preflight(scenes, validation=validation, calibration=calibration)
        base_identity, reward_status = checks["base_model_identity"], checks["reward_status"]
        root = Path(config.output_dir)
        root.mkdir(parents=True, exist_ok=False)
        checkpoint = Path(resume) if resume else None
        if checkpoint and read_json(checkpoint / "manifest.json")["run_signature"] != self.run_signature:
            raise ValueError("resume configuration/data/reward/source-adapter mismatch")
        self._load(checkpoint / "policy" if checkpoint else None)
        if checkpoint:
            self._restore(checkpoint)
        write_json(
            root / "run.json",
            {
                "config": asdict(config),
                "run_signature": self.run_signature,
                "reward_contract": self.reward.contract,
                "reward_status": reward_status,
                "environment": {name: version(name) for name in ("torch", "transformers", "peft")},
                "reference_sha256": self.reference_sha,
                "base_model_identity": base_identity,
                "resume_from": str(checkpoint) if checkpoint else None,
                "claim": "PPO with external AI feedback and length/empty penalties; no human-RLHF effectiveness claim",
            },
        )
        status, last_checkpoint = "completed", checkpoint
        try:
            if validation:
                self._evaluate(validation, root / f"validation-{self.update:06d}")
            for update_index in range(self.update, config.updates):
                selected = random.Random(config.seed + update_index).choices(scenes, k=config.rollout_batch_size)
                segments, traces = [], []
                for index, scene in enumerate(selected):
                    batch, trace = self._collect_episode(scene, update_index * config.rollout_batch_size + index)
                    segments.extend(batch)
                    traces.append(trace)
                write_jsonl(root / f"rollouts-{update_index + 1:06d}.jsonl", traces)
                if not segments:
                    status = "stopped_no_scorable_rollouts"
                    break
                token_count = sum(item["old_log_probs"].numel() for item in segments)
                reference_kl = (
                    sum(item["reference_kl"] * item["old_log_probs"].numel() for item in segments) / token_count
                )
                if reference_kl > config.max_reference_kl:
                    status = "stopped_reference_kl"
                    break
                metrics = self._optimize(segments)
                self.update = update_index + 1
                metrics.update(
                    update=self.update,
                    reference_kl=reference_kl,
                    mean_turn_reward=sum(item["score"] for item in segments) / len(segments),
                    episodes=len(traces),
                    abstained_episodes=sum(t["status"] == "abstained" for t in traces),
                )
                self.history.append(metrics)
                last_checkpoint = self._checkpoint(root)
                if validation and (self.update % config.eval_every == 0 or self.update == config.updates):
                    self._evaluate(validation, root / f"validation-{self.update:06d}")
                if stop_after is not None and self.update >= stop_after:
                    status = "stopped_at_requested_checkpoint"
                    break
            assert_reference_unchanged(self.model, self.reference_sha)
            self.model.save_pretrained(root / "adapter", selected_adapters=["default"])
            self.tokenizer.save_pretrained(root / "adapter")
            result = {
                "status": status,
                "updates_completed": self.update,
                "history": self.history,
                "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
                "reference_verified_unchanged": True,
                "reward_status": reward_status,
                "adapter_path": str(root / "adapter"),
                "persona_effectiveness": "not_evaluated",
            }
            write_json(root / "result.json", result)
            return result
        except Exception as exc:
            # No API error text or credentials in failure reports.
            write_json(
                root / "failure.json",
                {
                    "error_type": type(exc).__name__,
                    "updates_completed": self.update,
                    "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-scenes", type=Path, required=True)
    parser.add_argument("--validation-scenes", type=Path)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check data, model identity and initial token budgets without training or judge calls",
    )
    args = parser.parse_args()
    config = PPOConfig(**read_json(args.config))
    trainer = PersonaPPO(config, JudgeReward(read_json(args.judge_config)))
    inputs = dict(
        validation=read_jsonl(args.validation_scenes) if args.validation_scenes else None,
        calibration=read_json(args.calibration_report) if args.calibration_report else None,
    )
    scenes = read_jsonl(args.train_scenes)
    result = (
        trainer.preflight(scenes, **inputs)
        if args.check_only
        else trainer.train(scenes, **inputs, resume=args.resume, stop_after=args.stop_after)
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
