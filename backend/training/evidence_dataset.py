"""Auditable evidence-conditioned SFT/DPO export, without generating labels.

Rows must already be reviewed. The caller supplies the trusted system prompt;
row text can only enter untrusted user/history/assistant areas. Every variant
of a source group stays in the same split. Evidence text hashes also detect
cross-split reuse even when an annotator changes an evidence ID.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from inference.prompt_policy import build_grounded_user_message

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = "contextual-evidence-v1"
SPLITS = frozenset({"train", "validation", "test"})


def is_evidence_record(record: Mapping[str, Any]) -> bool:
    metadata = record.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("schema_version") == SCHEMA_VERSION


def validate_training_partitions(train, validation, *, packing: bool = False) -> None:
    """Protect exported split boundaries at the actual trainer entry point."""
    contextual_train = [row for row in train if is_evidence_record(row)]
    contextual_validation = [row for row in validation if is_evidence_record(row)]
    if not contextual_train and not contextual_validation:
        return
    if packing:
        raise ValueError("evidence-conditioned training requires packing=false to preserve complete contexts")
    if not contextual_train or not contextual_validation:
        raise ValueError("evidence-conditioned training requires fixed train and validation partitions")
    owners = {}
    for split, records in (("train", contextual_train), ("validation", contextual_validation)):
        for record in records:
            metadata = record["metadata"]
            if metadata.get("split") != split or metadata.get("review_status") != "approved":
                raise ValueError("unapproved or incorrectly assigned evidence training partition")
            group = _text(metadata.get("source_group"), "source_group")
            sources = _strings(metadata.get("source_ids"), "source_ids")
            evidence_hashes = _strings(metadata.get("evidence_text_sha256"), "evidence_text_sha256", allow_empty=True)
            if any(
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in evidence_hashes
            ):
                raise ValueError("invalid evidence content fingerprint")
            for key in [
                ("group", group),
                *(("source", source) for source in sources),
                *(("evidence_text", value) for value in evidence_hashes),
            ]:
                if owners.setdefault(key, split) != split:
                    raise ValueError("cross-split evidence provenance leakage at training entry")


def validate_preference_contexts(pairs, tokenizer, *, max_length: int, max_prompt_length: int) -> None:
    """Evidence preferences may not silently lose prompt facts in DPO/ORPO."""
    for pair in pairs:
        if not is_evidence_record(pair):
            continue
        metadata = pair["metadata"]
        if metadata.get("split") != "train" or metadata.get("review_status") != "approved":
            raise ValueError("only the approved train partition may enter preference optimization")
        prompt, chosen, rejected = pair.get("prompt"), pair.get("chosen"), pair.get("rejected")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(prompt[-1], Mapping)
            or prompt[-1].get("role") != "user"
        ):
            raise ValueError("evidence preference prompt must be conversational and end with user")
        start = int(isinstance(prompt[0], Mapping) and prompt[0].get("role") == "system")
        for index, message in enumerate(prompt):
            expected_role = "system" if index < start else ("user" if (index - start) % 2 == 0 else "assistant")
            if not isinstance(message, Mapping) or message.get("role") != expected_role:
                raise ValueError(
                    "evidence preference history must alternate user/assistant after an optional system prompt"
                )
            _text(message.get("content"), "prompt.content")
        for completion in (chosen, rejected):
            if (
                not isinstance(completion, list)
                or len(completion) != 1
                or not isinstance(completion[0], Mapping)
                or completion[0].get("role") != "assistant"
            ):
                raise ValueError("evidence preferences require one assistant completion")
            _text(completion[0].get("content"), "completion.content")
            # Use the tokenizer's normal conversational template, as TRL does.
            # Count the complete prompt+completion, not just assistant targets.
            ids = tokenizer.apply_chat_template([*prompt, *completion], tokenize=True, add_generation_prompt=False)
            if len(ids) > max_length:
                raise ValueError("evidence preference exceeds max_length; increase budget or curate context")
        if chosen == rejected:
            raise ValueError("chosen and rejected responses must differ")
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) > max_prompt_length:
            raise ValueError("evidence preference exceeds max_prompt_length; evidence truncation is forbidden")


def load_preference_training_rows(path: Path) -> list[dict[str, Any]]:
    """Accept conversational evidence exports without coercing them to strings.

    The legacy review-tool schema remains unchanged. Mixing conversational and
    plain-text records is rejected before Arrow/TRl can lose structure or fail.
    """
    from training.preference_data_schema import PreferencePair

    records = []
    formats = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError("preference row must be an object")
        if is_evidence_record(row):
            if row["metadata"].get("review_status") != "approved" or row["metadata"].get("split") != "train":
                raise ValueError("only approved contextual train preferences may be loaded")
            if not all(isinstance(row.get(field), list) for field in ("prompt", "chosen", "rejected")):
                raise ValueError("contextual preferences must retain conversational messages")
            records.append(dict(row))
            formats.add("conversational")
        else:
            pair = PreferencePair(**row)
            if pair.review_status == "approved":
                records.append(pair.to_jsonl_dict())
                formats.add("text")
    if len(formats) > 1:
        raise ValueError("train conversational and text preference formats in separate runs")
    return records


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _strings(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{field} must be a list")
    result = tuple(_text(item, field) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicates")
    return result


def text_fingerprint(value: str) -> str:
    return hashlib.sha256(" ".join(value.split()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceTrainingBundle:
    sft: tuple[dict[str, Any], ...]
    preference: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def build_evidence_training_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str,
    max_evidence_chars: int = 12000,
) -> EvidenceTrainingBundle:
    """Validate the WHOLE corpus before returning any exportable examples.

    Source/group IDs belong in metadata only, not supervision prompts. Expected
    evidence IDs validate annotations but never mark the answer inside context.
    No silent context truncation: a training target must not rely on omitted text.
    """
    system_prompt = _text(system_prompt, "system_prompt")
    if isinstance(max_evidence_chars, bool) or max_evidence_chars < 1:
        raise ValueError("max_evidence_chars must be positive")
    seen_ids: set[str] = set()
    ownership: dict[tuple[str, str], str] = {}
    sft: list[dict[str, Any]] = []
    preference: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    split_counts = {split: 0 for split in sorted(SPLITS)}
    for row in rows:
        if row.get("schema_version") != SCHEMA_VERSION or row.get("review_status") != "approved":
            raise ValueError("only approved contextual-evidence-v1 rows can be exported")
        key = _text(row.get("id"), "id")
        if key in seen_ids:
            raise ValueError(f"duplicate record ID: {key}")
        seen_ids.add(key)
        split = _text(row.get("split"), "split")
        if split not in SPLITS:
            raise ValueError("unknown split")
        group = _text(row.get("source_group"), "source_group")
        sources = _strings(row.get("source_ids"), "source_ids")
        provenance = [("group", group), *(("source", source) for source in sources)]
        query = _text(row.get("query"), "query")
        answer = _text(row.get("answer"), "answer")
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        evidence_ids: set[str] = set()
        blocks: dict[str, list[str]] = {"knowledge": [], "memory": []}
        for item in evidence:
            if not isinstance(item, Mapping):
                raise ValueError("invalid evidence")
            evidence_id = _text(item.get("id"), "evidence.id")
            if evidence_id in evidence_ids:
                raise ValueError("duplicate evidence ID")
            evidence_ids.add(evidence_id)
            source_id = _text(item.get("source_id"), "evidence.source_id")
            if source_id not in sources:
                raise ValueError("evidence source must appear in source_ids")
            kind = item.get("kind")
            if kind not in blocks:
                raise ValueError("evidence kind must be memory or knowledge")
            content = _text(item.get("content"), "evidence.content")
            blocks[kind].append(f"[{evidence_id}]\n{content}")
            provenance.append(("evidence_text", text_fingerprint(content)))
        expected = _strings(row.get("supporting_evidence_ids"), "supporting_evidence_ids", allow_empty=True)
        if not set(expected) <= evidence_ids:
            raise ValueError("supporting evidence ID is not present in context")
        answerable = row.get("answerable")
        if not isinstance(answerable, bool) or answerable != bool(expected):
            raise ValueError("answerability and supporting evidence disagree")
        # Counterfactual variants inherit the original scene group; never split
        # them randomly. Source IDs and evidence hashes provide extra checks.
        for provenance_key in provenance:
            owner = ownership.setdefault(provenance_key, split)
            if owner != split:
                raise ValueError(f"cross-split provenance leakage: {provenance_key[0]}")
        knowledge = "\n\n".join(blocks["knowledge"])
        memory = "\n\n".join(blocks["memory"])
        if len(knowledge) > max_evidence_chars or len(memory) > max_evidence_chars:
            raise ValueError("evidence exceeds budget; curate the example instead of silently truncating")
        history = row.get("history", [])
        if not isinstance(history, list):
            raise ValueError("history must be a list")
        messages = [{"role": "system", "content": system_prompt}]
        for index, entry in enumerate(history):
            if not isinstance(entry, Mapping) or entry.get("role") != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("history must consist of complete user/assistant pairs")
            messages.append({"role": entry["role"], "content": _text(entry.get("content"), "history.content")})
        if len(history) % 2:
            raise ValueError("history must end with assistant")
        messages.append(
            {
                "role": "user",
                "content": build_grounded_user_message(
                    query,
                    knowledge,
                    max_chars=max_evidence_chars,
                    memory_context=memory,
                ),
            }
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source_group": group,
            "source_ids": list(sources),
            "evidence_text_sha256": sorted({value for kind, value in provenance if kind == "evidence_text"}),
            "split": split,
            "assistant_supervision": "last",
            "supporting_evidence_ids": list(expected),
            "require_full_context": True,
            "answerable": answerable,
            "review_status": "approved",
        }
        sft.append({"id": key, "messages": [*messages, {"role": "assistant", "content": answer}], "metadata": metadata})
        rejected = row.get("rejected_answer")
        if rejected is not None:
            rejected = _text(rejected, "rejected_answer")
            if rejected == answer:
                raise ValueError("chosen and rejected responses must differ")
            preference.append(
                {
                    "id": key,
                    "prompt": messages,
                    "chosen": [{"role": "assistant", "content": answer}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                    "metadata": dict(metadata),
                }
            )
        split_counts[split] += 1
        source_counts[group] = source_counts.get(group, 0) + 1
    canonical = json.dumps(sft, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    canonical_preferences = json.dumps(preference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return EvidenceTrainingBundle(
        tuple(sft),
        tuple(preference),
        {
            "schema_version": SCHEMA_VERSION,
            "records": len(sft),
            "preference_pairs": len(preference),
            "split_counts": split_counts,
            "source_group_counts": source_counts,
            "system_prompt_sha256": text_fingerprint(system_prompt),
            "sft_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "preference_sha256": hashlib.sha256(canonical_preferences.encode("utf-8")).hexdigest(),
            "fingerprint_format": "ordered-record-array; sorted-keys; compact-json; utf-8; ensure_ascii=false",
            "quality_status": "schema_validated_not_model_evaluated",
        },
    )
