"""Execute necessary-condition reads without inverting implication direction.

Only whole self-owned requests with verbatim condition/action atoms qualify.
This does not predict behavior, infer today's circumstances, or grant permission.
"""
from __future__ import annotations

import re

from character.conditional_memory import verified_constraint_atoms


def _verified_rule(item):
    if (item.memory_type != 'user_fact' or item.historical or item.valid_to
            or item.status not in {'active', 'current'} or item.confidence < .8
            or not item.source_message_ids):
        return None
    qualifiers = dict(item.qualifiers)
    atoms = verified_constraint_atoms(qualifiers, item.content, item.evidence)
    if atoms is None:
        return None
    from character.conditional_memory import NecessaryCondition

    if item.memory_key != NecessaryCondition(atoms[0][0], atoms[1]).memory_key:
        return None
    return atoms


def _read_kind(message: str, condition: str, action: str) -> str:
    action = re.escape(action)
    condition = re.escape(condition)
    quoted_condition = rf'(?:{condition}|“{condition}”|"{condition}")'
    ending = r'[？?。]?'
    read = rf'(?:请问)?我{action}(?:需要什么条件|的必要条件是什么|有什么限制){ending}'
    entailment = (
        rf'(?:如果|只要){condition}[，,]我(?:就)?一定{action}吗{ending}|'
        rf'仅凭{quoted_condition}(?:就)?能确定我{action}吗{ending}|'
        rf'{condition}就意味着我一定{action}吗{ending}'
    )
    if re.fullmatch(read, message.strip()):
        return 'read'
    if re.fullmatch(entailment, message.strip()):
        return 'not_sufficient'
    return ''


def render_constraint_response(message, context):
    if (context is None or getattr(context, 'memory_status', 'retrieval_error') == 'retrieval_error'
            or getattr(context, 'branch_context', '')):
        return None
    packets = getattr(context, 'memory_packets', ())
    admitted_ids = getattr(context, 'used_memory_ids', ())
    candidates = []
    for item in packets:
        if item.memory_id not in admitted_ids:
            continue
        rule = _verified_rule(item)
        if rule is not None:
            conditions, action = rule
            for condition in conditions:
                kind = _read_kind(message, condition, action)
                if kind:
                    candidates.append((conditions, action, kind))
    if len(set(candidates)) != 1:
        return None
    conditions, action, kind = candidates[0]
    # A logical shortcut must not bypass the existing safety response path.
    # Classify the requested action as well as the full question: conditional
    # phrasing can otherwise hide first-person risk from the ordinary gate.
    from character.situation_analyzer import SITUATION_SAFETY, SituationAnalyzer

    analyzer = SituationAnalyzer()
    if any(analyzer.analyze(text)[0] == SITUATION_SAFETY for text in (message, '我' + action)):
        return None
    # Other admitted constraints on this action cannot be ignored just because
    # their condition did not match the premise in the question.
    for item in packets:
        qualifiers = dict(item.qualifiers)
        if (item.memory_id in admitted_ids and qualifiers.get('action') == action
                and _verified_rule(item) != (conditions, action)):
            return None
    if len(conditions) > 1:
        listed = '、'.join(f'“{condition}”' for condition in conditions)
        lead = '你之前明确补充过' if kind == 'read' else '不能仅凭这一点确定。你之前明确补充过'
        return f'{lead}，“{action}”需要同时满足{listed}；这些都是必要条件，即使满足也不表示行动一定发生。'
    condition = conditions[0]
    if kind == 'read':
        return f'你之前说过，只有“{condition}”才会“{action}”。这是必要条件，不表示满足它就一定会这样做。'
    return f'不能仅凭这一点确定。你说的是只有“{condition}”才会“{action}”；满足这个必要条件，不代表行动一定发生。'
