"""A shared view of a leading memory-management wrapper; sources stay intact."""

import re

_LEADING_REMEMBER = re.compile(r'^(?:请)?(?:记住|记下)(?:一下)?[，,：:\s]*')
_LEADING_CORRECTION = re.compile(r'^(?:更正|纠正)(?:一下)?[，,：:]\s*')


def memory_statement_body(message: str) -> str:
    """Remove at most one affirmative leading wrapper, never a later clause.

    This is not evidence admission or authorization. Consumers must validate
    the full remaining statement, its owner, conditions and temporal scope.
    """
    text = message.strip()
    corrected = _LEADING_CORRECTION.sub('', text, count=1)
    return corrected if corrected != text else _LEADING_REMEMBER.sub('', text, count=1)
