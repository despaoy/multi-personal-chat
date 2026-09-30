"""Validation shared by preference freezing and the actual trainer entry point."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def fingerprint(value) -> str:
    def normalize(item):
        if isinstance(item, str):
            return "".join(item.split()).casefold()
        if isinstance(item, list):
            return [normalize(part) for part in item]
        if isinstance(item, dict):
            return {key: normalize(part) for key, part in item.items()}
        return item

    payload = json.dumps(normalize(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reviewed_pair_digest(row):
    metadata = row["metadata"]
    value = {key: row[key] for key in ("id", "prompt", "chosen", "rejected")}
    value["provenance"] = {
        key: metadata[key]
        for key in (
            "persona",
            "source_group",
            "source_ids",
            "evidence_text_sha256",
            "review_sha256",
            "chosen_candidate_id",
            "rejected_candidate_id",
        )
    }
    if "preference_family" in metadata:
        value["provenance"]["preference_family"] = metadata["preference_family"]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def provenance_keys(row) -> set[tuple[str, str]]:
    metadata = row.get("metadata") or {}
    keys = {("prompt", fingerprint(row["prompt"]))}
    if row.get("id"):
        keys.add(("id", str(row["id"])))
    for field in ("source_group", "scene_id"):
        value = metadata.get(field)
        if value:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            keys.add((field, value))
    for field in ("source_ids", "evidence_text_sha256"):
        values = metadata.get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{field} must be a list of nonempty strings")
        keys.update((field, value) for value in values)
    return keys


def validate_pairs(rows, *, split: str) -> None:
    if split not in {"train", "validation"}:
        raise ValueError("only train/validation may be used during optimization")
    if not rows:
        raise ValueError(f"empty {split} preference dataset")
    seen_ids, seen_prompts, formats = set(), {}, set()
    family_prompts = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        if row.get("mock") is True or metadata.get("mock") is True:
            raise ValueError("mock preferences may not enter real training")
        if metadata.get("schema_version") == "persona-preference-v1":
            if metadata.get("human_final_approved") is not True or not metadata.get("review_sha256"):
                raise ValueError("persona preferences require bound human review")
            if metadata.get("pair_content_sha256") != reviewed_pair_digest(row):
                raise ValueError("reviewed preference content/provenance binding mismatch")
        statuses = (row.get("review_status"), metadata.get("review_status"))
        if "approved" not in statuses:
            raise ValueError("explicit approved review status is required")
        for status in statuses:
            if status is not None and status != "approved":
                raise ValueError("only approved preferences may enter optimization")
        for declared in (row.get("split"), metadata.get("split")):
            if declared is not None and declared != split:
                raise ValueError(f"incorrect preference partition: expected {split}, got {declared}")
        fields = [row.get(key) for key in ("prompt", "chosen", "rejected")]
        if all(isinstance(value, str) and value.strip() for value in fields):
            formats.add("text")
        elif all(isinstance(value, list) and value for value in fields):
            formats.add("conversational")
            prompt, chosen, rejected = fields
            start = int(isinstance(prompt[0], dict) and prompt[0].get("role") == "system")
            for index, message in enumerate(prompt):
                role = "system" if index < start else ("user" if (index - start) % 2 == 0 else "assistant")
                if not isinstance(message, dict) or message.get("role") != role:
                    raise ValueError("preference history must alternate user/assistant after optional system")
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError("empty preference message")
            if prompt[-1]["role"] != "user":
                raise ValueError("preference prompt must end with user")
            for completion in (chosen, rejected):
                if (
                    len(completion) != 1
                    or not isinstance(completion[0], dict)
                    or completion[0].get("role") != "assistant"
                    or not isinstance(completion[0].get("content"), str)
                    or not completion[0]["content"].strip()
                ):
                    raise ValueError("one nonempty assistant completion is required")
        else:
            raise ValueError("prompt/chosen/rejected must be nonempty and have the same format")
        if fingerprint(row["chosen"]) == fingerprint(row["rejected"]):
            raise ValueError("chosen and rejected must differ")
        prompt_key = fingerprint(row["prompt"])
        family = metadata.get("preference_family")
        if family is not None:
            if (metadata.get("schema_version") != "persona-preference-v1"
                or not isinstance(family, dict)
                or set(family) != {"id", "max_negatives", "review_method"}
                or not isinstance(family.get("id"), str) or not family["id"].strip()
                or type(family.get("max_negatives")) is not int
                or not 2 <= family["max_negatives"] <= 4
                or family.get("review_method") != "individually_reviewed_distinct_negatives"
                or not row.get("id") or not metadata.get("source_group")
                or not metadata.get("source_ids") or not metadata.get("rejected_candidate_id")):
                raise ValueError("invalid reviewed multi-negative preference family")
            previous_prompt = family_prompts.setdefault(family["id"], prompt_key)
            if previous_prompt != prompt_key:
                raise ValueError("preference family must have one prompt")
        if prompt_key in seen_prompts:
            siblings = seen_prompts[prompt_key]
            first = siblings[0]
            previous = first.get("metadata") or {}
            if family is None or family != previous.get("preference_family"):
                raise ValueError("duplicate normalized preference prompt")
            for field in ("source_group", "source_ids", "evidence_text_sha256", "chosen_candidate_id"):
                if metadata.get(field) != previous.get(field):
                    raise ValueError("preference family provenance must agree")
            if fingerprint(row["chosen"]) != fingerprint(first["chosen"]):
                raise ValueError("preference family chosen response must agree")
            if len(siblings) >= family["max_negatives"]:
                raise ValueError("too many negatives in preference family")
            if any(fingerprint(row["rejected"]) == fingerprint(s["rejected"])
                   or metadata["rejected_candidate_id"] == s["metadata"]["rejected_candidate_id"]
                   for s in siblings):
                raise ValueError("duplicate preference family negative")
            siblings.append(row)
        else:
            seen_prompts[prompt_key] = [row]
        if row.get("id"):
            if str(row["id"]) in seen_ids:
                raise ValueError("duplicate preference id")
            seen_ids.add(str(row["id"]))
        provenance_keys(row)
    if len(formats) != 1:
        raise ValueError("train conversational and text preference formats in separate runs")


def validate_partitions(train, validation=None) -> None:
    validate_pairs(train, split="train")
    if validation is None:
        return
    validate_pairs(validation, split="validation")
    if isinstance(train[0]["prompt"], list) != isinstance(validation[0]["prompt"], list):
        raise ValueError("train and validation must use the same preference format")
    training_keys = set().union(*(provenance_keys(row) for row in train))
    for row in validation:
        overlap = training_keys & provenance_keys(row)
        if overlap:
            raise ValueError(f"cross-split preference leakage: {sorted(key[0] for key in overlap)}")


def validate_adapter_path(path: str) -> None:
    if not path:
        return
    root = Path(path)
    if not root.is_dir() or not (root / "adapter_config.json").is_file():
        raise ValueError("SFT adapter path must contain adapter_config.json; refusing base-model fallback")
    if not any((root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("SFT adapter path has no adapter weights")
    config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("bias", "none") != "none":
        raise ValueError("preference training requires a LoRA adapter with bias=none")


def validate_token_budgets(rows, tokenizer, *, max_length, max_prompt_length, render_conversation=None):
    """Do not let TRL silently truncate the character, evidence or conversation."""
    if not 0 < max_prompt_length < max_length:
        raise ValueError("require 0 < max_prompt_length < max_length")
    for row in rows:
        if isinstance(row["prompt"], list) and render_conversation is not None:
            row = render_conversation({key: row[key] for key in ("prompt", "chosen", "rejected")}, tokenizer)
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            full = [
                tokenizer.apply_chat_template(
                    [*prompt, *row[key]],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                for key in ("chosen", "rejected")
            ]
        else:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = []
            for key in ("chosen", "rejected"):
                # DPO tokenizes the prompt and completion independently; ORPO also
                # inspects the combined string. Budget for the larger of both.
                separate = prompt_ids + tokenizer(row[key], add_special_tokens=False)["input_ids"]
                combined = tokenizer(prompt + row[key], add_special_tokens=False)["input_ids"]
                full.append(separate if len(separate) >= len(combined) else combined)
            # TRL may add BOS/EOS to plain text; reserve those positions.
            prompt_ids = [0, *prompt_ids]
            full = [[0, *ids, 0] for ids in full]
        if len(prompt_ids) > max_prompt_length:
            raise ValueError("preference exceeds max_prompt_length; silent context truncation is forbidden")
        if any(len(ids) > max_length for ids in full):
            raise ValueError("preference exceeds max_length; silent completion truncation is forbidden")


def group_split(rows, *, seed=42, validation_fraction=0.2):
    """Keep transitive source/scene/evidence groups together; never silently row-split."""
    validate_pairs(rows, split="train")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    ordered = sorted(rows, key=lambda row: (str(row.get("id", "")), fingerprint(row["prompt"])))
    parents = list(range(len(ordered)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, row in enumerate(ordered):
        metadata = row.get("metadata") or {}
        if not (metadata.get("source_group") or metadata.get("source_ids") or metadata.get("scene_id")):
            raise ValueError("grouped split requires source_group, source_ids or scene_id for every pair")
        for key in sorted(provenance_keys(row)):
            if key in owners:
                parents[root(index)] = root(owners[key])
            else:
                owners[key] = index
    groups = {}
    for index, row in enumerate(ordered):
        groups.setdefault(root(index), []).append(row)
    groups = list(groups.values())
    if len(groups) < 2:
        raise ValueError("at least two independent source groups are required")
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(rows) * validation_fraction))
    validation, train = [], []
    for index, group in enumerate(groups):
        if len(validation) < target and index < len(groups) - 1:
            validation.extend(group)
        else:
            train.extend(group)

    # Return copies with explicit partition metadata; never rewrite the input asset.
    def assign(items, split):
        result = []
        for row in items:
            copied = dict(row)
            if "split" in copied:
                copied["split"] = split
            copied["metadata"] = {**(row.get("metadata") or {}), "split": split}
            result.append(copied)
        return result

    train, validation = assign(train, "train"), assign(validation, "validation")
    validate_partitions(train, validation)
    return train, validation
