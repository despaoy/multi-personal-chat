"""Literal self-alias predicates, independent of the model's name label."""

import re

_NAME = r"[\w\u4e00-\u9fff]{1,12}"
_POSITIVE = re.compile(r"我(?:的)?(?:常用)?(?:别名|昵称)(?:是|叫)(?P<name>" + _NAME + r")")
_EXCLUDED = re.compile(r"(?P<name>" + _NAME + r")(?:不是|不属于)(?:我(?:的)?|本人(?:的)?)(?:别名|昵称)")
_FOLLOWING_LIMIT = re.compile(r"^(?:但|不过|然而|只有|仅|除非|暂时|可能|也许|计划|将来|明年)")


def explicit_alias_value(text, value):
    return any(match["name"] == value for match in _POSITIVE.finditer(text))


def identity_alias_projection(*, source, evidence, value):
    """Require a complete, literal source predicate, not a clipped name token.

    A negative self-ownership predicate can retain its contiguous explanation
    about a friend; it grants no positive friend or self-name assertion.
    """
    if not evidence or evidence not in source:
        return None
    sentences = re.findall(r"[^。！？!?；;\n]+", source)
    matches = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence not in evidence:
            continue
        if index + 1 < len(sentences) and _FOLLOWING_LIMIT.match(sentences[index + 1].strip()):
            continue
        positive = _POSITIVE.fullmatch(sentence)
        excluded = _EXCLUDED.fullmatch(sentence)
        if positive and value == positive["name"]:
            name = positive["name"]
            matches.append(("user_fact", "user_alias", f"用户明确说自己的常用别名是{name}", 0.7))
        elif excluded and (value == excluded["name"] or value in evidence and excluded["name"] in value):
            name = excluded["name"]
            matches.append(("user_fact", "user_alias_exclusion_" + name, f"用户明确说{name}不是本人的别名", 0.6))
    return matches[0] if len(matches) == 1 else None


def normalize_alias_labels(raw, *, projection):
    if projection is None or not isinstance(raw, dict):
        return raw
    # The independent complete declarative predicate supplies admission.
    # All actual conditions/unknown labels still face ordinary validation.
    return {
        key: value
        for key, value in raw.items()
        if key != "certainty" or value not in ("稳定", "stable", "明确", "明确立即生效")
    }
