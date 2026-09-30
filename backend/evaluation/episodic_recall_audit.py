"""Offline raw-evidence recall comparison; not a production memory reader.

No model calls, database access, fact extraction, or current-state inference.
Scope is an exact pre-normalized tuple supplied by the experiment. This does
not establish authorization for a future database adapter.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from character.memory_extractor import memory_write_allowed


@dataclass(frozen=True)
class Episode:
    source_id: str
    scope: tuple[str, ...]
    session: str
    timestamp: str
    message: str
    reply: str = ''


def _terms(text: str) -> set[str]:
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    return {text[i:i+2] for i in range(len(text)-1)}


def packet_text(episodes: list[Episode]) -> str:
    return json.dumps([asdict(row) for row in episodes], ensure_ascii=False, separators=(',', ':'))


def select_evidence(episodes: list[Episode], query: str, scope: tuple[str, ...], *,
                    mode: str, max_chars: int, additional_hit_ids: tuple[str, ...] = ()) -> dict:
    if mode not in {'turn', 'session', 'suffix'} or max_chars < 2:
        raise ValueError('Expected turn/session/suffix mode and budget sufficient for an empty JSON array')
    rows = [row for row in episodes if row.scope == scope]
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError('Ambiguous duplicate source IDs')
    extra_hits = set(additional_hit_ids)
    if not extra_hits <= {row.source_id for row in rows}:
        raise ValueError('Additional hits must belong to the scoped source pool')
    # Excluding a prohibited correction but retaining its original statement
    # would manufacture stale evidence. In this experiment exclude that session.
    blocked = {row.session for row in rows if (row.message and not memory_write_allowed(row.message))
               or (row.reply and not memory_write_allowed(row.reply))}
    # Experimental upper-bound window: all subsequent available source rows,
    # not a claim that any later utterance actually updates the matched event.
    suffix_rows = sorted(rows, key=lambda row: (row.timestamp, row.source_id))
    rows = [row for row in rows if row.session not in blocked]
    groups = defaultdict(list)
    for row in rows:
        groups[row.source_id if mode == 'turn' else row.session].append(row)
    terms = _terms(query)
    candidates = []
    for key, group in groups.items():
        # Replies may supply context but cannot create a lexical retrieval hit.
        score = max((max(len(terms & _terms(row.message)), int(row.source_id in extra_hits))
                     for row in group), default=0)
        if score:
            group.sort(key=lambda row: (row.timestamp, row.source_id))
            candidates.append((score, key, group))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if mode == 'suffix':
        hits = [i for i, row in enumerate(suffix_rows)
                if terms & _terms(row.message) or row.source_id in extra_hits]
        candidates = []
        if hits:
            first_session = suffix_rows[hits[0]].session
            start = next(i for i, row in enumerate(suffix_rows) if row.session == first_session)
            window = suffix_rows[start:]
            if not any(row.session in blocked for row in window):
                candidates = [(1, 'complete_suffix', window)]
    selected: list[Episode] = []
    budget_rejected = []
    for _, key, group in candidates:
        trial = sorted([*selected, *group], key=lambda row: (row.timestamp, row.source_id))
        if len(packet_text(trial)) > max_chars:
            budget_rejected.append(key)
            continue
        selected = trial
    return dict(mode=mode, source_ids=[row.source_id for row in selected],
                packet=packet_text(selected), chars=len(packet_text(selected)),
                eligible_groups=len(candidates), budget_rejected=budget_rejected,
                blocked_sessions=sorted(blocked))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.suite.read_text(encoding='utf-8'))
    results = []
    for case in cases:
        scope = tuple(case['scope'])
        rows = [Episode(**{**row, 'scope': tuple(row['scope'])}) for row in case['episodes']]
        for budget in case['budgets']:
            for mode in ('turn', 'session', 'suffix'):
                result = select_evidence(rows, case['query'], scope, mode=mode, max_chars=budget)
                result.update(case=case['id'], budget=budget,
                    missing_required_evidence=sorted(set(case['required_evidence']) - set(result['source_ids'])),
                    answer_quality='not_evaluated', synthetic=True)
                results.append(result)
    with args.output.open('x', encoding='utf-8') as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(json.dumps([dict(case=r['case'], mode=r['mode'], budget=r['budget'], ids=r['source_ids'],
                           missing=r['missing_required_evidence']) for r in results], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
