"""History/current comparison from native complete observed sources."""

import json
import re
from html import unescape

from evaluation.memory_correction_audit import audit_memory_correction_wire


def has_never_received_assertion(reply):
    """Distinguish an assertion from an explicit local denial of that claim."""
    claim = re.compile(r'(?:从未|从来(?:都)?没有|一直没(?:有)?).{0,10}(?:收到|确认)')
    denied = re.compile(r'(?:不是(?:(?:在|要)?(?:说|写成|表示|指))?|并非|不代表|'
                        r'并不意味着|不能(?:说|写成)|不该(?:说|写成))\s*(?:你|我|用户|本人)?\s*$')
    for clause in re.split(r'[，,。！？；;\n]', reply):
        for match in claim.finditer(clause):
            if not denied.search(clause[:match.start()]):
                return True
    return False


def audit_memory_history_wire(proof, calls, fixture):
    checks = audit_memory_correction_wire(proof, calls, fixture)
    final = proof['generation'][-1]
    answers = [c for c in calls[slice(*final['cloud_call_range'])] if c['request'].get('max_tokens') == 1024]
    if len(answers) != 1:
        return {'one_actual_history_comparison_answer': False}
    wire = unescape(answers[0]['request']['messages'][-1]['content'])
    block = re.search(r'<character_memory[^>]*>\n(.*?)\n</character_memory>', wire, re.S)
    packets = [json.loads(line[2:]) for line in block[1].splitlines() if line.startswith('- {')] if block else []
    first, second = [next((p for p in packets if c['message'] in p.get('evidence', [])), None)
                     for c in fixture['cases'][:2]]
    reply = final['response'].get('reply', '')
    checks.update(
        complete_initial_observation_admitted=first is not None,
        complete_later_correction_admitted=second is not None,
        observed_source_clocks_preserve_sequence=bool(first and second and first.get('observed_at')
                                                    and second.get('observed_at') and first['observed_at'] < second['observed_at']),
        both_sources_retain_quote_semantics=bool(first and second) and all(
            p.get('content_semantics') == 'quoted_source' and p.get('speaker_role') == 'user'
            and p.get('subject_scope') == 'not_resolved' and p.get('temporal_mode') == 'observation'
            and not p.get('valid_from') for p in (first, second)),
        historical_confirmation_not_erased=bool(re.search(r'(?:当时|先前|最初|以前|曾经).{0,45}(?:已经|已|确实|收到).{0,10}(?:收到|书面|确认)', reply)),
        not_rewritten_as_never_received=not has_never_received_assertion(reply),
        old_sources_and_answers_not_replayed=proof.get('reused_native_fixture', {}).get('source_writing_generations_replayed') == 0
        and proof['reused_native_fixture'].get('prior_answer_generations_replayed') == 0 and len(proof['generation']) == 1,
        existing_documents_not_reimported=proof.get('document_imports_replayed') == 0,
        copied_index_matches_native_source=bool(proof.get('reused_native_fixture', {}).get('source_vectors_sha256')),
    )
    return checks
