"""Execute a complete current correction without claiming a completed write."""
from __future__ import annotations

from character.conditional_memory import parse_necessary_condition
from character.memory_extractor import extract_memories
from character.situation_analyzer import SITUATION_SAFETY, SituationAnalyzer
from knowledge.task_dependency import local_context_only


def render_current_constraint_response(message: str, context) -> str | None:
    if context is None or getattr(context, 'branch_context', '') or not local_context_only(message):
        return None
    # Full-message match prevents a correction from swallowing another task.
    # Reuse assertion gates: quoted, fictional and private/unsaved statements
    # must not be silently treated as an ordinary correction command.
    rule = parse_necessary_condition(message)
    if rule is None or rule.operation != 'replace':
        return None
    items = extract_memories(message)
    if (len(items) != 1 or items[0].operation != 'replace'
            or items[0].evidence != message.strip()):
        return None
    analyzer = SituationAnalyzer()
    if any(analyzer.analyze(text)[0] == SITUATION_SAFETY for text in (message, '我' + rule.action)):
        return None
    return (f'按你这次的更正，“{rule.action}”的必要条件是“{rule.condition}”；'
            '满足这个条件，也不表示行动一定发生。')
