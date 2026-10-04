"""Closed other-owner reads may reject candidates before semantic review.

Ambiguous coreference is a review task, not an authorization decision. This
module grants no ownership and never rewrites the complete original query.
"""

import re

from character.memory_subject import OWNER_PATTERN, SELF_OWNERS

_OWNER = r"(?P<owner>" + OWNER_PATTERN.pattern + r")"
_LOOKUPS = (
    ("name", re.compile(_OWNER + r"(?:的)?(?:名字|姓名)(?:是(?:什么|谁)|叫什么)?")),
    ("name", re.compile(_OWNER + r"叫(?:什么|什么名字|啥)")),
    (
        "preference",
        re.compile(
            _OWNER + r"(?:本人|自己)?(?:平时|通常|最|比较|目前|现在)*"
            r"(?:喜欢|讨厌|偏好|钟意)(?:什么|哪些|啥)"
        ),
    ),
    ("preference", re.compile(_OWNER + r"的(?:偏好|爱好)(?:是什么|有哪些)?")),
)


def closed_other_subject_fields(query: str) -> frozenset[str]:
    """Require every original clause to be a complete other-owner lookup.

    Unknown clauses, self reads, quoted material and output instructions defer
    to the mandatory contextual selector. No nearest-pronoun verdict is used.
    """
    clauses = [part for part in re.split(r"[。！？!?；;\n]", query) if part.strip()]
    fields = set()
    for clause in clauses:
        text = "".join(clause.split())
        text = re.sub(r"^(?:请问|请告诉我|告诉我)", "", text)
        text = re.sub(r"(?:吗|呢)$", "", text)
        found = next(((field, pattern.fullmatch(text)) for field, pattern in _LOOKUPS if pattern.fullmatch(text)), None)
        if found is None or found[1]["owner"] in SELF_OWNERS:
            return frozenset()
        fields.add(found[0])
    return frozenset(fields)
