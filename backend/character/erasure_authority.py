"""Bind mixed erase/retain instructions to existing scoped logical objects.

This is admission, not retrieval ranking or an alternate semantic writer. An
unresolved retained object denies mixed mutations; a model cannot authorize it.
"""

import re
from dataclasses import dataclass

_NEGATIVE = re.compile(r"(?:不要|别|不许|不能|不用|无需|不必|不想|不希望).{0,16}(?:删掉|删除|清除|移除|忘掉|忘记)")
_CONNECTORS = r"(?:(?:请|也|但|但是|同时|另外|而|务必|另一方面)\s*)*"
_PROTECTED = re.compile(
    "^" + _CONNECTORS + r"(?:"
    r"(?:不要|别|不许|不能|不用|无需|不必|不想|不希望)(?:再)?"
    r"(?:忘掉|彻底忘记|忘记|删除|删掉|清除|移除|从(?:长期)?记忆(?:里面|中|里)?(?:彻底)?(?:删除|清除|删掉|移除))"
    r"|(?:继续)?保留)(?P<object>.+)$"
)
_ARCHIVE = re.compile(r"聊天历史|聊天记录|对话历史|对话记录")


def _normalized(text):
    return re.sub(r"\s+", "", str(text)).casefold()


def _object(record):
    key = str(record.get("memory_key") or "")
    value = key.partition("_")[2]
    return _normalized(value) if len(value) >= 2 else ""


def _identity(record):
    return str(record.get("id") or record.get("memory_id") or "")


@dataclass(frozen=True)
class PartialErasurePlan:
    valid: bool
    affirmative_clauses: tuple[str, ...]
    protected_clauses: tuple[str, ...]
    allowed_ids: tuple[str, ...] = ()
    protected_ids: tuple[str, ...] = ()
    protected_keys: tuple[str, ...] = ()
    protected_objects: tuple[str, ...] = ()
    unresolved_protection: bool = False

    def accepts_erasure(self, proposal):
        if not self.valid or self.unresolved_protection or proposal.target_memory_id not in self.allowed_ids:
            return False
        evidence = _normalized(proposal.evidence)
        # Positive target matching above is independent of a model's chosen
        # evidence. A clipped negative command cannot become authorization.
        if len(evidence) < 8 or _NEGATIVE.search(proposal.evidence):
            return False
        return any(
            _normalized(clause) in evidence or evidence in _normalized(clause)
            for clause in self.affirmative_clauses
            if evidence
        )

    def protects_mutation(self, proposal):
        if proposal.target_memory_id in self.protected_ids:
            return True
        memory = proposal.memory
        if memory is None:
            return False
        if memory.memory_key in self.protected_keys:
            return True
        content = _normalized(memory.content)
        return any(value in content for value in self.protected_objects)

    def model_constraints(self):
        return dict(
            allowed_erase_memory_ids=list(self.allowed_ids),
            protected_memory_ids=list(self.protected_ids),
            affirmative_clauses=list(self.affirmative_clauses),
            protected_clauses=list(self.protected_clauses),
            unresolved_protection=self.unresolved_protection,
            erase_source_ids_allowed=False,
        )


def partial_erasure_plan(message, records=None):
    """Resolve named logical objects only; never use scores or model IDs.

    No plan means an ordinary request retains its existing admission. A plan
    with no allowed targets or unresolved protection authorizes no mutation.
    The caller retains the entire original message for semantic interpretation.
    """
    from character.memory_llm import _ARCHIVE_ONLY_ERASURE_NEGATION, _ERASE_REQUEST_PATTERN
    from character.quoted_erasure_authority import quoted_erasure_plan

    quoted = quoted_erasure_plan(message, records)
    if quoted is not None:
        return quoted
    text = re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"', "", message or "").strip()
    if re.search(r"^(?:如果|假如|假设|要是)|(?:他说|她说|朋友说|你说过)", text):
        return None
    affirmative, protected = [], []
    invalid_negative = False
    for match in re.finditer(r"[^，,。；;！？!?\n]+(?:[，,。；;！？!?\n]|$)", text):
        raw = match.group()
        clause = raw.rstrip("，,。；;！？!?\n").strip()
        if _ARCHIVE_ONLY_ERASURE_NEGATION.fullmatch(clause):
            continue
        retention = _PROTECTED.fullmatch(clause)
        if retention:
            # A qualification/combined target mentioning the archive is not
            # the precise archive-only exception and cannot relax admission.
            if _ARCHIVE.search(retention.group("object")):
                invalid_negative = True
            else:
                protected.append(clause)
        elif _NEGATIVE.search(clause):
            invalid_negative = True
        elif _ERASE_REQUEST_PATTERN.search(clause):
            if raw.endswith(("?", "？")) or re.search(r"如果|假如|假设|要是", clause):
                invalid_negative = True
            else:
                affirmative.append(clause)
    if not affirmative:
        return None
    if invalid_negative:
        return PartialErasurePlan(False, tuple(affirmative), tuple(protected))
    if not protected:
        return None
    if records is None:
        return PartialErasurePlan(True, tuple(affirmative), tuple(protected))
    records = tuple(records)
    kept = []
    unresolved_clauses = []
    for clause in protected:
        matched = [record for record in records if _object(record) and _object(record) in _normalized(clause)]
        if not matched:
            unresolved_clauses.append(clause)
        kept.extend(matched)
    # A repeated named preference reference can point only to an already
    # independently matched, unique protected preference. Unknown/new objects
    # and generic record references still cannot relax protection.
    unresolved = False
    known = {_identity(record): record for record in kept}
    for clause in unresolved_clauses:
        obj = _PROTECTED.fullmatch(clause).group("object")
        pointer = re.fullmatch(r"(?:这|那)(?:个|一)?(?:项|条)?(?P<topic>.*?)偏好", obj)
        preferences = [
            record for record in known.values() if str(record.get("memory_key") or "").startswith("preference_")
        ]
        if (
            pointer is None
            or len(preferences) != 1
            or (pointer.group("topic") and _normalized(pointer.group("topic")) not in _object(preferences[0]))
        ):
            unresolved = True
    protected_ids = tuple(dict.fromkeys(_identity(record) for record in kept))
    allowed = tuple(
        dict.fromkeys(
            _identity(record)
            for record in records
            if _identity(record) not in protected_ids
            and _object(record)
            and any(_object(record) in _normalized(clause) for clause in affirmative)
        )
    )
    return PartialErasurePlan(
        True,
        tuple(affirmative),
        tuple(protected),
        () if unresolved else allowed,
        protected_ids,
        tuple(dict.fromkeys(str(record.get("memory_key") or "") for record in kept)),
        tuple(dict.fromkeys(_object(record) for record in kept)),
        unresolved,
    )
