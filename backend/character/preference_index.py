"""Source-grounded short preference indexes for conditional display labels."""

import re

from character.quoted_erasure_authority import masked_quotes

_LABEL = re.compile(r"(?P<head>[^（()）]{1,48})(?:（(?P<cn>[^（()）]+)）|\((?P<ascii>[^（()）]+)\))")
_NECESSARY = re.compile(r"^(?:仅在|仅当|只有|仅)")
_STATUS = r"(?:已提交|已完成|审核通过|校验通过|确认通过|核验通过|通过|已确认|已登记)"
_ATOM = re.compile(rf"[^，,。！？!?；;和且或]+{_STATUS}")


def conditional_preference_label(value):
    match = _LABEL.fullmatch("".join(str(value or "").split()))
    return match if match and _NECESSARY.match(match["cn"] or match["ascii"]) else None


def _same_condition(left, right):
    if left == right:
        return True
    # Only complete, literal status atoms admit equivalent AND renderings.
    # OR, negation, added/dropped atoms and words inside entities cannot supply
    # an approximate semantic match. Unknown grammar keeps its literal gate.
    parts = [re.split(r"并且|以及|和|且", text) for text in (left, right)]
    return parts[0] == parts[1] and len(parts[0]) > 1 and all(_ATOM.fullmatch(atom) for atom in parts[0])


def grounded_preference_index(value, *, kind, evidence, source_message, qualifiers, target_key=""):
    """Strip a necessary-condition label only with the complete literal claim.

    This changes an index's rendering, not the predicate, qualifiers, source or
    lifecycle. Unknown formats, ungrounded conditions and quoted declarations
    supply no authority. The caller retains ordinary qualifier/target checks.
    """
    match = conditional_preference_label(value)
    if not match or kind not in {"like", "dislike"} or not isinstance(qualifiers, dict):
        return None
    head = match["head"]
    if target_key and target_key != "preference_" + head[:20]:
        return None
    condition = qualifiers.get("condition")
    if not isinstance(condition, str):
        return None
    condition = "".join(condition.split())
    source_condition = re.fullmatch(
        rf"只有(?P<body>[^，,。！？!?；;]+)才(?:使用|选择|饮用|办理){re.escape(head)}", condition
    )
    label_condition = _NECESSARY.sub("", match["cn"] or match["ascii"], count=1)
    if label_condition.endswith("时"):
        label_condition = label_condition[:-1]
    if not source_condition or not _same_condition(label_condition, source_condition["body"]):
        return None
    predicate = "我喜欢" if kind == "like" else "我不喜欢"
    claim = predicate + head + "且" + condition
    try:
        source = "".join(masked_quotes(source_message)[0].split())
        quote = "".join(masked_quotes(evidence)[0].split())
    except ValueError:
        return None
    if claim not in source or claim not in quote:
        return None
    return head
