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
    group_modes: tuple[tuple[str, ...], ...] = ()

    @property
    def atom_groups(self):
        return tuple(tuple(SourceReadAtom(value, self.group_modes[i][j] if self.group_modes else self.match_mode)
                           for j, value in enumerate(group)) for i, group in enumerate(self.groups))

    @property
    def selectors(self):
        return tuple(dict.fromkeys(atom for group in self.atom_groups for atom in group))

    def matches_fragment(self, fragment, body):
        if self.match_mode == 'mixed':
            raise ValueError('Mixed source dependencies require their individual matcher')
        return SourceReadAtom(fragment, self.match_mode).matches(body)

    def matches(self, body):
        return any(all(atom.matches(body) for atom in group) for group in self.atom_groups)


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


@dataclass(frozen=True)
class SourceReadAtom:
    value: str
    match_mode: str

    def matches(self, body):
        return (identifier_match(self.value, body) if self.match_mode == 'identifier_token'
                else self.value in body)


_LITERAL_PREFIX = rf'{_PREFIX}(?:查找|检索|读取|查看)\s*(?:包含|含有)\s*'
_LITERAL_SUFFIX = re.compile(r'\s*的\s*(?:完整|全部)?(?:用户)?(?:原话|原始发言)(?:记录)?\s*(?:[。；;\n]|$)')


def _literal_at(view, quotes, position, valid_fragment):
    index = next((i for i, quote in enumerate(quotes) if quote[0] >= position), len(quotes))
    if index == len(quotes) or not re.fullmatch(_LITERAL_PREFIX, view[position:quotes[index][0]]):
        if re.match(_LITERAL_PREFIX, view[position:]):
            raise ValueError('Unclosed literal dependency')
        return None
    members, operator = [], None
    while index < len(quotes):
        _, z, fragment = quotes[index]
        if (not fragment.strip() or '\n' in fragment or not valid_fragment(fragment)
                or any(char in fragment for char in '“”‘’「」『』"')):
            raise ValueError('Invalid literal dependency')
        members.append(fragment)
        index += 1
        read = _LITERAL_SUFFIX.match(view, z)
        if read:
            atoms = tuple(SourceReadAtom(value, 'literal_substring') for value in dict.fromkeys(members))
            return (tuple((atom,) for atom in atoms) if operator == 'or' else (atoms,)), read.end()
        if index == len(quotes):
            raise ValueError('Unclosed literal dependency')
        connector = view[z:quotes[index][0]].strip()
        next_operator = ('and' if connector in {'和', '与', '及', '并且'} else
                         'or' if connector in {'或', '或者'} else None)
        if next_operator is None or (operator is not None and next_operator != operator):
            raise ValueError('Ambiguous literal dependency')
        operator = next_operator
    raise ValueError('Unclosed literal dependency')


def source_read_plan(query, valid_fragment, unresolved_tail):
    """Union fully closed reads while preserving every member's own matcher.

    A literal substring and an identifier token with the same spelling are
    different dependencies. Same-record literal conjunctions remain intact.
    No unfinished additional read can authorize just the completed prefix.
    """
    from character.quoted_erasure_authority import masked_quotes

    if not isinstance(query, str):
        return SourceReadPlan()
    try:
        view, quotes = masked_quotes(query)
        position, groups, identifiers = 0, [], []
        while position < len(view):
            literal = _literal_at(view, quotes, position, valid_fragment)
            if literal is not None:
                added, position = literal
                groups.extend(added)
                continue
            match = next((m for pattern in _FRAMES if (m := pattern.match(view, position))), None)
            if match is None:
                break
            members = re.findall(_IDENTIFIER, match['ids'])
            if not all(any(char.isdigit() for char in member) for member in members):
                return SourceReadPlan()
            identifiers.extend(members)
            groups.extend((SourceReadAtom(member, 'identifier_token'),) for member in members)
            position = match.end()
    except ValueError:
        return SourceReadPlan()
    if not groups:
        return SourceReadPlan()
    remainder = view[position:]
    declared = {atom.value for group in groups for atom in group}
    later = re.findall(rf'(?<![A-Za-z0-9_-]){_IDENTIFIER}(?![A-Za-z0-9_-])', remainder)
    if (unresolved_tail(remainder)
            or re.search(rf'{_ACTION}{_SELECTOR}|按{_SELECTOR}', remainder)
            or (identifiers and any(any(char.isdigit() for char in value) and value not in declared for value in later))
            or re.search(r'(?:不要|别|禁止|不必|无需|排除|不包括)\s*(?:再|分别|逐项)?'
                         r'(?:核对|核查|读取|查看|比较|资料|原话|原始|记录|编号)', remainder)
            or re.search(r'(?:核对|核查|比较|对照)[^。；;\n]*(?:其他|另一|新的|不同)'
                         r'[^。；;\n]*(?:原话|原始|记录|资料|目录|版本)', remainder)):
        return SourceReadPlan()
    groups = tuple(dict.fromkeys(groups))
    modes = {atom.match_mode for group in groups for atom in group}
    return SourceReadPlan(tuple(tuple(atom.value for atom in group) for group in groups),
                          next(iter(modes)) if len(modes) == 1 else 'mixed',
                          tuple(tuple(atom.match_mode for atom in group) for group in groups))
