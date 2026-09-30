"""Shared whole-turn character budgeting after database scope filtering."""
from collections.abc import Iterable


def history_turn_limit(limit: int) -> int:
    """Bound database loading without silently reducing cloud's 128-turn view.

    This counts stored question/answer turns, not individual model messages.
    Character and final model-token budgets remain independent safeguards.
    """
    return max(1, min(int(limit), 500))


def assemble_history(turns_newest_first: Iterable[tuple[str | None, str | None]], max_chars: int) -> list[dict[str, str]]:
    """Return a chronological contiguous suffix of complete stored turns.

    Nonpositive budgets preserve the existing unlimited convention. A turn
    that cannot fit ends selection; never skip it to resurrect older context.
    Legacy single-sided rows remain single-sided, not newly orphaned replies.
    """
    kept: list[list[dict[str, str]]] = []
    total = 0
    for message, reply in turns_newest_first:
        turn = [{'role': role, 'content': content.strip()}
                for role, content in (('user', message), ('assistant', reply))
                if content and content.strip()]
        size = sum(len(item['content']) for item in turn)
        if max_chars > 0 and total + size > max_chars:
            break
        total += size
        kept.append(turn)
    return [message for turn in reversed(kept) for message in turn]
