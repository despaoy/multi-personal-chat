"""Offline literal quotation provenance audit; never a semantic truth judge.

Only indexed raw evidence (not summaries, viewpoint metadata or prior replies)
can establish a labeled speaker. Narration is never assigned to a narrator by
guessing the first-person viewpoint. Unmatched/paraphrased quotations remain
unknown rather than being called fabricated.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_DIALOGUE = re.compile(r'^\s*\[([^\]\n]{1,40})\]\s*「([^」\n]+)」\s*$')
_CLAIM = re.compile(
    r'(?P<speaker>[\u4e00-\u9fffA-Za-z]{1,24}?)(?:也曾|曾经|曾|也|则)?'
    r'(?:说道|说|提到|回应|回答)(?:过)?\s*[：:]\s*[“「](?P<quote>[^”」\n]+)[”」]'
)


def audit_quote_attribution(reply: str, documents: Sequence[Mapping], *,
                            aliases: Mapping[str, str] | None = None) -> list[dict]:
    """Return evidence-limited findings, retaining every matching source line.

    Aliases must be supplied explicitly; no suffix-based name equivalence.
    Source offsets are usable only when complete raw evidence line counts
    agree with the indexed inclusive span. Otherwise local offsets are kept.
    """
    aliases = aliases or {}
    lines = []
    speakers = set(aliases)
    for document in documents:
        content = document.get('content', '')
        if not isinstance(content, str) or '\n证据：' not in content:
            continue
        raw_lines = content.split('\n证据：', 1)[1].splitlines()
        source = document.get('source') or {}
        start, end = source.get('line_start'), source.get('line_end')
        located = type(start) is int and type(end) is int and start > 0 and end - start + 1 == len(raw_lines)
        for offset, line in enumerate(raw_lines):
            dialogue = _DIALOGUE.fullmatch(line)
            speaker = dialogue[1] if dialogue else None
            if speaker:
                speakers.add(speaker)
            lines.append(dict(text=dialogue[2] if dialogue else line, speaker=speaker,
                              unresolved_structure=not dialogue and any(mark in line for mark in ('[', '「', '」')),
                              document_id=document.get('id'), source_path=source.get('source_path'),
                              line=start + offset if located else None, evidence_offset=offset))

    findings = []
    for claim in _CLAIM.finditer(reply):
        prefix = claim['speaker']
        # Strip only grammatical response prefixes; don't invent a short-name alias.
        prefix = re.sub(r'^(?:而|不过|但是)', '', prefix)
        speaker = aliases.get(prefix, prefix)
        quote = claim['quote']
        matches = [line for line in lines if quote in line['text']]
        labeled = {aliases.get(line['speaker'], line['speaker']) for line in matches if line['speaker']}
        if not matches:
            status = 'unmatched_literal'
        elif speaker in labeled:
            status = 'supported_label'
        elif labeled and prefix in speakers:
            status = 'conflicting_label'
        elif not labeled and any(line['unresolved_structure'] for line in matches):
            status = 'unresolved_structure'
        elif not labeled:
            status = 'narration_only'
        else:
            status = 'unknown_speaker'
        findings.append(dict(claimed_speaker=prefix, quote=quote, status=status,
                             labeled_speakers=sorted(labeled), evidence=matches))
    return findings


def main():
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    counts = Counter()
    for path in args.trace:
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            retrieval = row.get('generation', {}).get('plan', {}).get('retrieval', {})
            reply = row.get('response', {}).get('reply', '')
            findings = audit_quote_attribution(reply, retrieval.get('documents', []))
            counts.update(item['status'] for item in findings)
            records.append(dict(trace=str(path), case_id=row['case_id'], turn=row['turn'],
                                findings=findings, assessment='literal_provenance_only_not_semantic_pass'))
    with args.output.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(dict(turns=len(records), counts=dict(counts)), ensure_ascii=False))


if __name__ == '__main__':
    main()
