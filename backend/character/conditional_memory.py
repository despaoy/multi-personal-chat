"""Bounded declarative constraints, not predictions or a free-text logic solver.

For 'I do A only if C', A implies C; C never implies A. Keep the
verbatim atoms and provenance rather than guessing synonyms or current states.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class NecessaryCondition:
    condition: str
    action: str
    operation: str = 'assert'

    @property
    def memory_key(self) -> str:
        # Stable per action: changing its restriction must use version handling.
        return 'constraint_' + sha256(self.action.encode('utf-8')).hexdigest()[:24]

    def qualifiers(self, evidence: str) -> tuple[tuple[str, str], ...]:
        return (('context', evidence), ('kind', 'necessary_condition'),
                ('condition', self.condition), ('action', self.action))


def same_necessary_condition(left: object, right: object) -> bool:
    """Compare validated atoms, not source typography or correction markers.

    Unknown qualifiers must keep the conservative conflict path: ignoring an
    extra temporal/exception slot would silently merge different restrictions.
    """
    rules = []
    for qualifiers in (left, right):
        if (not isinstance(qualifiers, dict)
                or set(qualifiers) != {'kind', 'condition', 'action', 'context'}
                or qualifiers.get('kind') != 'necessary_condition'
                or not isinstance(qualifiers.get('context'), str)):
            return False
        rule = parse_necessary_condition(qualifiers['context'])
        if (rule is None or rule.condition != qualifiers['condition']
                or rule.action != qualifiers['action']):
            return False
        rules.append((rule.condition, rule.action))
    return rules[0] == rules[1]


def parse_necessary_condition(text: str) -> NecessaryCondition | None:
    """Single self-owned habitual assertion; caller applies privacy/fiction gates."""
    text = text.strip()
    correction = re.match(r'^(更正|纠正|补充)(?:一下)?[，,:：]\s*', text)
    operation = ('append' if correction[1] == '补充' else 'replace') if correction else 'assert'
    if correction:
        text = text[correction.end():]
    match = re.fullmatch(r'我只有([^，,。！？!?；;才\n]{1,40})才([^，,。！？!?；;\n]{1,40})[。]?', text)
    if not match:
        return None
    condition, action = (x.strip() for x in match.groups())
    # Unknown ownership, nested implications, and transient plans are not rules.
    if (not condition or not action or re.search(r'只有|如果|除非|才|今天|明天|这次|这回', condition + action)
            or re.match(r'你|他|她|它|我们|你们|他们|她们|朋友|同事|爸爸|妈妈', action)
            or re.search(r'吗|呢|么|可能|也许|大概|似乎|好像', condition + action)):
        return None
    return NecessaryCondition(condition, action, operation)


def constraint_set_content(action: str, conditions: tuple[str, ...]) -> str:
    return '用户自述的必要条件：' + action + '；各项都必要：' + '、'.join(f'“{c}”' for c in conditions)


def verified_constraint_atoms(qualifiers: object, content: str, evidence: object):
    """Recover a single rule or explicit additive chain from original evidence.

    No inference from summaries. Every added atom must have an explicit additive
    assertion; a changed ordinary assertion cannot be silently conjoined.
    """
    if not isinstance(qualifiers, dict) or not isinstance(evidence, (list, tuple)) or not evidence:
        return None
    if qualifiers.get('kind') == 'necessary_condition':
        if set(qualifiers) != {'kind', 'condition', 'action', 'context'}:
            return None
        for quote in evidence:
            if not isinstance(quote, str):
                continue
            rule = parse_necessary_condition(quote)
            if (rule and content == '用户自述：' + quote
                    and dict(rule.qualifiers(quote)) == qualifiers):
                return (rule.condition,), rule.action
        return None
    if (qualifiers.get('kind') != 'necessary_condition_set'
            or set(qualifiers) != {'kind', 'conditions', 'action', 'context'}):
        return None
    rules = [parse_necessary_condition(q) if isinstance(q, str) else None for q in evidence]
    if (any(r is None for r in rules) or any(r.operation != 'append' for r in rules[1:])
            or any(r.action != rules[0].action for r in rules)):
        return None
    conditions = tuple(dict.fromkeys(r.condition for r in rules))
    action = rules[0].action
    if (len(conditions) < 2 or qualifiers['action'] != action
            or qualifiers['conditions'] != json.dumps(conditions, ensure_ascii=False)
            or content != constraint_set_content(action, conditions) or qualifiers['context'] != content):
        return None
    return conditions, action
