"""Shared explicit owner grammar for memory questions and candidate evidence.

This recognizes direct subjects, not arbitrary coreference or entailment.
Unknown syntax is not a positive self-ownership verdict.
"""
import re

# Keep question routing and write admission on the same owner vocabulary.
OWNER_PATTERN = re.compile(
    r'我(?:的)?(?:朋友|同学|室友|同事|老师|家人|父母|父亲|母亲|爸爸|妈妈|哥哥|姐姐|弟弟|妹妹)|'
    r'我们|咱们|你们|他们|她们|我|咱|你|您|他|她'
)
SELF_OWNERS = frozenset({'我', '我们', '咱', '咱们'})


def is_source_observation(row: dict) -> bool:
    """Read the application-owned observation envelope, not a model actor label."""
    metadata = row.get('metadata')
    return (isinstance(metadata, dict) and metadata.get('content_semantics') == 'quoted_source'
            and metadata.get('speaker_role') == 'user'
            and metadata.get('described_subject') == 'not_resolved')


_CLAUSES = re.compile(r'[^，,。！？!?；;\n]+')
_DIRECT_SUBJECT = re.compile(
    r'^(?:但是|不过|而|但)?\s*(?P<owner>' + OWNER_PATTERN.pattern + r')'
    r'(?:本人|自己)?(?:的)?(?:目前|现在|以前|曾经|去年|今年|最近|已经|刚刚|刚|一直|正在|也|很|不|并不|最|挺|比较)*'
    r'(?P<predicate>不喜欢|喜欢|讨厌|爱|叫|名字是|专业是|读的是|学的是|来自|(?:老家|故乡|家乡)(?:在|是)|居住在|住在|搬(?:家)?到|在|是|准备|工作)'
)


def explicitly_other_subject(*, source: str, evidence: str, value: str) -> bool:
    """Reject a value only when all matching source clauses explicitly belong elsewhere.

    Inspect the original clause, not just a model-trimmed evidence quote.
    Multiple occurrences/unknown subjects cannot establish this verdict.
    No text is rewritten; callers retain the full source independently.
    """
    if not evidence or not value:
        return False
    positions = [m.start() for m in re.finditer(re.escape(evidence), source)]
    if not positions:
        return False
    subjects = []
    for start in positions:
        end = start + len(evidence)
        for clause in _CLAUSES.finditer(source):
            if clause.start() >= end or clause.end() <= start:
                continue
            overlap = source[max(start, clause.start()):min(end, clause.end())]
            if value not in overlap:
                continue
            match = _DIRECT_SUBJECT.match(clause.group().strip())
            subjects.append(match['owner'] if match else None)
    return bool(subjects) and all(owner is not None and owner not in SELF_OWNERS for owner in subjects)
