"""Remove redundant preference labels only with complete self assertion proof."""

import re

from character.memory_extractor import complete_self_assertions


def normalize_preference_labels(raw, *, kind, evidence, source_message, canonical):
    """Labels cannot broaden a quote or substitute for real qualifications."""
    if not isinstance(raw, dict) or kind not in {"like", "dislike"} or not canonical:
        return raw

    def surface(text):
        return text.strip().rstrip("。！？!?；;").strip()

    quote = surface(evidence)
    sentences = re.findall(r"[^。！？!?；;\n]+(?:[。！？!?；;]|$)", source_message)
    for sentence in sentences:
        if surface(sentence) != quote:
            continue
        assertions = complete_self_assertions(sentence)
        if (
            len(assertions) != 1
            or assertions[0].qualifiers
            or assertions[0].memory_type != "user_fact"
            or assertions[0].memory_key != canonical[1]
            or assertions[0].content != canonical[2]
        ):
            continue
        redundant = {("certainty", "稳定"), ("preference_strength", "否定" if kind == "dislike" else "肯定")}
        return {key: value for key, value in raw.items() if not isinstance(value, str) or (key, value) not in redundant}
    return raw
