"""Pinned, read-only Chinese role benchmark adapter; no training/registry import.

Benchmark IDs, book labels and applicable evaluation metrics remain outside
model prompts. Named dialogue speakers and the intended role profile are input
evidence, not evaluation labels. Malformed dialogues fail instead of being fixed
by guessing a speaker or dropping a turn.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

SOURCE_SHA256 = {
    "data/test_data.jsonl": "6d9723a1484f1d2739bcf388c7960abe1933b7a5a1fddae0170cc9ea7c66dcd8",
    "data/character_profiles.json": "d9b4c9f7ab63b89b3c348ae88b43c87872f0f6a5cb06d7f31cbe64a630e6434e",
    "data/id2metric.jsonl": "df80ee716124a7deacc6f24ee372a4600b719cdaa79f01cefcdfb286359f1801",
}
SUBSET_SEED = "charactereval-contextual-pilot-v1"


@dataclass(frozen=True)
class PublicCharacterCase:
    case_id: str
    role: str
    novel_name: str
    profile: str
    history: tuple[dict[str, str], ...]
    query: str
    metric_ids: tuple[str, ...]


def load_pinned_corpus(cache_dir):
    values = {}
    for name, expected in SOURCE_SHA256.items():
        raw = (cache_dir / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"pinned public benchmark hash differs: {name}")
        values[name] = json.loads(raw.decode("utf-8-sig"))
    return values["data/test_data.jsonl"], values["data/character_profiles.json"], values["data/id2metric.jsonl"]


def stable_character_subset(rows, *, roles=8, per_role=2):
    if type(roles) is not int or type(per_role) is not int or min(roles, per_role) < 1:
        raise ValueError("subset counts must be positive integers")
    groups = {}
    seen = set()
    for row in rows:
        key = str(row["id"])
        if key in seen:
            raise ValueError("duplicate benchmark case ID")
        seen.add(key)
        role = row["role"]
        if not isinstance(role, str) or not role.strip():
            raise ValueError("missing role name")
        groups.setdefault(role, []).append(row)

    def digest(value):
        return hashlib.sha256(f"{SUBSET_SEED}:{value}".encode()).digest()

    return [
        row
        for role in sorted(groups, key=digest)[:roles]
        for row in sorted(groups[role], key=lambda item: digest(str(item["id"])))[:per_role]
    ]


def adapt_character_case(row, profiles, metric_map):
    role = row["role"]
    profile = profiles.get(role)
    # Official profiles are structured dictionaries, even though the upstream
    # ChatGLM script embeds their Python repr in a string. Preserve every field
    # in deterministic JSON rather than dropping attributes or guessing prose.
    if isinstance(profile, dict) and profile:
        profile = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("missing textual or structured role profile")
    context = row.get("context")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("empty dialogue context")
    messages = []
    for line in context.splitlines():
        speaker, separator, utterance = line.partition("：")
        if not separator or not speaker.strip() or not utterance.strip():
            raise ValueError("dialogue line must retain an explicit named speaker and utterance")
        messages.append({"role": "assistant" if speaker.strip() == role else "user", "content": line})
    if messages[-1]["role"] != "user":
        raise ValueError("benchmark continuation must follow another speaker")
    metrics = metric_map.get(str(row["id"]), [])
    if not isinstance(metrics, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(label, str) for label in pair)
        for pair in metrics
    ):
        raise ValueError("invalid evaluator-only metric labels")
    return PublicCharacterCase(
        str(row["id"]),
        role,
        str(row.get("novel_name", "")),
        profile,
        tuple(messages[:-1]),
        messages[-1]["content"],
        tuple(pair[0] for pair in metrics),
    )


def dialogue_messages(case):
    """Model-facing inputs cannot accidentally include metric or benchmark IDs."""
    return [dict(message) for message in case.history] + [{"role": "user", "content": case.query}]
