"""Explicit source dependencies identify speech; they never certify its facts."""
import re
from dataclasses import dataclass


def identifier_pattern(identifier):
    return r'(^|[^A-Za-z0-9_-])' + re.escape(identifier) + r'($|[^A-Za-z0-9_-])'


def identifier_match(identifier, body):
    return (isinstance(identifier, str) and bool(identifier) and isinstance(body, str)
            and bool(re.search(identifier_pattern(identifier), body)))


@dataclass(frozen=True)
class SourceReadPlan:
    groups: tuple[tuple[str, ...], ...] = ()
    match_mode: str = 'not_resolved'

    def matches_fragment(self, fragment, body):
        return (identifier_match(fragment, body) if self.match_mode == 'identifier_token'
                else fragment in body)

    def matches(self, body):
        return any(all(self.matches_fragment(fragment, body) for fragment in group)
                   for group in self.groups)


_IDENTIFIER = r'[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*'
_SELECTOR = r'(?:条目|资料)?编号\s*[:：]?\s*'
_ACTION = r'(?:分别|逐项|逐一)?(?:核对|核查|读取|查看|比较|对照)\s*'
_PREFIX = r'\s*(?:(?:再|另|另外|同时)?(?:请)?|请(?:再|另|另外|同时)?)'
_IDS = rf'(?P<ids>{_IDENTIFIER}(?:\s*(?:、|，|,|和|与|及|以及)\s*{_IDENTIFIER})*)'
_ORIGINAL = r'(?:完整|全部)?(?:原始资料|原始数据|原始记录|原始发言|原话记录|原话)\s*(?:[。；;，,\n]|$)'
_FRAMES = (
    re.compile(rf'{_PREFIX}{_ACTION}{_SELECTOR}{_IDS}\s*的\s*{_ORIGINAL}'),
    re.compile(rf'{_PREFIX}按{_SELECTOR}{_IDS}\s*{_ACTION}{_ORIGINAL}'),
)


def _unquoted_view(query):
    from character.quoted_erasure_authority import masked_quotes
    try:
        return masked_quotes(query)[0]
    except ValueError:
        return None


def has_identifier_read_selector(query):
    view = _unquoted_view(query)
    return view is not None and bool(re.search(rf'{_ACTION}{_SELECTOR}|按{_SELECTOR}', view))


def identifier_read_plan(query, unresolved_tail):
    """Each explicitly enumerated ID is an independent complete-source root.

    Distinct ID reads are unions, not a requirement to concatenate different
    records. All matching original versions remain eligible; unknown or
    contradictory additional requests do not authorize a partial plan.
    """
    view = _unquoted_view(query)
    if view is None:
        return SourceReadPlan()
    position, identifiers = 0, []
    while position < len(view):
        match = next((m for pattern in _FRAMES if (m := pattern.match(view, position))), None)
        if match is None:
            break
        members = re.findall(_IDENTIFIER, match['ids'])
        if not all(any(char.isdigit() for char in member) for member in members):
            return SourceReadPlan()
        identifiers.extend(members)
        position = match.end()
    if not identifiers:
        return SourceReadPlan()
    remainder = view[position:]
    # A detached unquoted identifier may denote another requested dependency.
    later = re.findall(rf'(?<![A-Za-z0-9_-]){_IDENTIFIER}(?![A-Za-z0-9_-])', remainder)
    if (any(any(char.isdigit() for char in value) and value not in identifiers for value in later)
            or unresolved_tail(remainder)
            or re.search(r'(?:不要|别|禁止|不必|无需|排除|不包括)\s*(?:再|分别|逐项)?'
                         r'(?:核对|核查|读取|查看|比较|资料|原话|原始|记录|编号)', remainder)
            or re.search(r'(?:核对|核查|比较|对照)[^。；;\n]*(?:其他|另一|新的|不同)'
                         r'[^。；;\n]*(?:原话|原始|记录|资料|目录|版本)', remainder)):
        return SourceReadPlan()
    return SourceReadPlan(tuple((identity,) for identity in dict.fromkeys(identifiers)), 'identifier_token')
