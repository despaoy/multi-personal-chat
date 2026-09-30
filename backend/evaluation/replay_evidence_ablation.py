"""Replay frozen model inputs; input ablations are diagnostic, never a pass label.

No retrieval, production writes, system-prompt edits or runtime monkeypatching.
The original question stays identical. Evidence is unchanged by default; the
optional card view uses complete evidence from the same ranked documents.
Removing assistant turns may remove follow-up referents, so differences are not
a general causal quality score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen


def variants(messages):
    if not messages or messages[-1].get('role') != 'user':
        raise ValueError('Expected a final user message containing the evidence')
    if not any(m.get('role') == 'assistant' for m in messages):
        raise ValueError('No assistant history to ablate')
    return {'full_history': messages,
            'without_assistant_history': [m for m in messages if m.get('role') != 'assistant']}


def replay_parameters(call, generation_plan):
    """The executed call wins over planning defaults; expose missing observations."""
    executed = call.get('parameters') or {}
    keys = ('temperature', 'max_tokens', 'top_p', 'repetition_penalty',
            'frequency_penalty', 'enable_thinking')
    assumed = [key for key in keys if executed.get(key) is None]
    values = {key: executed[key] if executed.get(key) is not None else generation_plan[key]
              for key in keys}
    return values, assumed


def card_evidence_messages(messages, documents, *, max_chars=6000):
    """Diagnostic evidence view from the same ranked cards; never synthesize text.

The stored card evidence is not freshly source-verified. Reject incomplete or
over-budget inputs instead of cherry-picking or clipping their qualifications.
"""
    blocks = []
    for doc in documents:
        content = doc.get('content', '')
        if '\n证据：' not in content:
            raise ValueError('Every selected card must carry its own evidence')
        evidence = content.split('\n证据：', 1)[1].strip()
        if not evidence:
            raise ValueError('Empty card evidence')
        blocks.append(f"【卡片关联证据】{doc['title']}\n{evidence}")
    text = '\n\n'.join(blocks)
    if not text or len(text) > max_chars:
        raise ValueError('Complete evidence does not fit the diagnostic budget')
    result = [dict(m) for m in messages]
    pattern = r'(?s)(<retrieved_evidence\b[^>]*>\n).*?(\n</retrieved_evidence>)'
    content, count = re.subn(pattern, lambda m: m[1] + text + m[2], result[-1]['content'])
    if count != 1:
        raise ValueError('Expected exactly one retrieved evidence block')
    result[-1]['content'] = content
    return result


def annotated_evidence_messages(messages, retrieval):
    """Add index annotations only; retain the exact admitted quote/background set."""
    from html import escape

    from knowledge.evidence_packet import render_card_evidence

    documents = {doc['id']: doc for doc in retrieval['documents']}
    blocks = []
    changed = 0
    packets = retrieval.get('evidence_packets') or ()
    if '\n\n'.join(p['text'] for p in packets) != retrieval['evidence']:
        raise ValueError('Recorded packets do not reconstruct the admitted evidence')
    for packet in packets:
        text = packet['text']
        if text.startswith('【卡片关联证据】'):
            ids = packet['document_ids']
            if len(ids) != 1 or ids[0] not in documents:
                raise ValueError('Evidence packet needs one known document')
            text = render_card_evidence(documents[ids[0]], text.split('\n', 1)[1])
            changed += 1
        blocks.append(text)
    if not changed:
        raise ValueError('No annotatable evidence packets')
    old = escape(retrieval['evidence'], quote=False)
    new = escape('\n\n'.join(blocks), quote=False)
    result = [dict(m) for m in messages]
    wrapped = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    original = wrapped + old + '\n</retrieved_evidence>'
    if result[-1]['content'].count(original) != 1:
        raise ValueError('Actual model input does not contain the recorded complete evidence')
    result[-1]['content'] = result[-1]['content'].replace(original, wrapped + new + '\n</retrieved_evidence>')
    return result


def speaker_evidence_messages(messages, retrieval):
    """Diagnostic line typing only; preserve every raw character and history.

    No narrator identity is inferred. Unknown/multiline quote syntax remains
    explicitly unclassified. This is not used by the production renderer.
    """
    from html import escape

    packets = retrieval.get('evidence_packets') or ()
    if '\n\n'.join(p['text'] for p in packets) != retrieval['evidence']:
        raise ValueError('Recorded packets do not reconstruct admitted evidence')
    documents = {doc['id']: doc for doc in retrieval['documents']}
    blocks = []
    changed = 0
    for packet in packets:
        text = packet['text']
        if text.startswith('【卡片关联证据】'):
            ids = packet['document_ids']
            if len(ids) != 1 or ids[0] not in documents:
                raise ValueError('Need exactly one known card')
            content = documents[ids[0]].get('content', '')
            if '\n证据：' not in content:
                raise ValueError('Missing raw card evidence')
            raw = content.split('\n证据：', 1)[1].strip()
            if not raw or not text.endswith(raw):
                raise ValueError('Raw evidence is not the admitted packet suffix')
            lines = []
            for line in raw.splitlines(keepends=True):
                if re.fullmatch(r'\s*\[[^\]\n]{1,40}\]\s*「[^」\n]+」\s*', line):
                    kind = '台词'
                elif any(mark in line for mark in ('[', '「', '」')):
                    kind = '未分类'
                else:
                    kind = '叙述'
                lines.append(f'【{kind}】' + line)
            text = text[:-len(raw)] + ''.join(lines)
            changed += 1
        blocks.append(text)
    if not changed:
        raise ValueError('No admitted raw cards')
    result = [dict(m) for m in messages]
    wrapped = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    original = wrapped + escape(retrieval['evidence'], quote=False) + '\n</retrieved_evidence>'
    if result[-1]['content'].count(original) != 1:
        raise ValueError('Actual model input differs from admitted evidence')
    replacement = wrapped + escape('\n\n'.join(blocks), quote=False) + '\n</retrieved_evidence>'
    result[-1]['content'] = result[-1]['content'].replace(original, replacement)
    return result


def without_background_messages(messages, retrieval):
    """Ablate only admitted background packets; keep full card evidence intact.

    Frozen system, history, question and card annotations stay unchanged. This
    is an offline comparison, not permission to remove context in production.
    """
    from html import escape

    packets = retrieval.get('evidence_packets') or ()
    if '\n\n'.join(packet['text'] for packet in packets) != retrieval['evidence']:
        raise ValueError('Recorded packets do not reconstruct admitted evidence')
    if any(packet.get('kind') not in {'evidence', 'background'} for packet in packets):
        raise ValueError('Unknown packet kind')
    retained = [packet['text'] for packet in packets if packet['kind'] == 'evidence']
    if not retained or len(retained) == len(packets):
        raise ValueError('Need both admitted evidence and background packets')
    result = [dict(message) for message in messages]
    wrapped = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    original = wrapped + escape(retrieval['evidence'], quote=False) + '\n</retrieved_evidence>'
    if result[-1]['content'].count(original) != 1:
        raise ValueError('Actual model input differs from admitted evidence')
    replacement = wrapped + escape('\n\n'.join(retained), quote=False) + '\n</retrieved_evidence>'
    result[-1]['content'] = result[-1]['content'].replace(original, replacement)
    return result


def full_task_messages(messages, trace):
    """Restore only the full user query in a frozen identity subtask input.

    History stays fixed; this isolates task wording, not the whole composite
    runtime policy. The known memory renderer is not scored by this replay.
    """
    from html import escape

    generation = trace['generation']
    residual = generation['plan']['retrieval'].get('identity_task', {}).get('query')
    original = trace.get('message')
    if (generation.get('response_mode') != 'task_composite'
            or not isinstance(residual, str) or not residual
            or not isinstance(original, str) or not original or original == residual):
        raise ValueError('Need a recorded composite with a distinct bound residual query')
    expected = '<user_query>\n' + escape(residual, quote=False) + '\n</user_query>'
    result = [dict(message) for message in messages]
    if not result or result[-1].get('role') != 'user' or not result[-1]['content'].endswith(expected):
        raise ValueError('Actual model input does not end with bound residual query')
    result[-1]['content'] = (result[-1]['content'][:-len(expected)] +
                             '<user_query>\n' + escape(original, quote=False) + '\n</user_query>')
    return result


def selection_evidence_messages(messages, trace, audit):
    """Compare a frozen retrieval result for the exact same complete query."""
    from html import escape

    if audit.get('query') != trace.get('message'):
        raise ValueError('Retrieval query must match the original complete query')
    retrieval = trace['generation']['plan']['retrieval']
    new = audit['result']
    for data, key in ((retrieval, 'evidence'), (new, 'context_text')):
        if not data.get('evidence_packets') or '\n\n'.join(p['text'] for p in data['evidence_packets']) != data[key]:
            raise ValueError('Packets must reconstruct the complete evidence')
    if not new['context_text'] or len(new['context_text']) > 6000:
        raise ValueError('Replacement exceeds the evidence budget')
    admitted = {i for p in new['evidence_packets'] for i in p['document_ids']}
    if admitted != {c['id'] for c in new['citations']}:
        raise ValueError('Replacement citations must match admitted evidence')
    prefix = '<retrieved_evidence trust="untrusted" purpose="factual_grounding">\n'
    old = prefix + escape(retrieval['evidence'], quote=False) + '\n</retrieved_evidence>'
    if not messages or messages[-1]['role'] != 'user' or messages[-1]['content'].count(old) != 1:
        raise ValueError('Actual model input differs from admitted evidence')
    result = [dict(m) for m in messages]
    result[-1]['content'] = result[-1]['content'].replace(
        old, prefix + escape(new['context_text'], quote=False) + '\n</retrieved_evidence>')
    return result


def select_trace(records, case, *, turn=None, memory_only=False):
    if memory_only == (turn is not None):
        raise ValueError('Choose one ordinary turn or memory-only probe')
    if turn is not None and turn < 1:
        raise ValueError('Ordinary turn must be positive')
    matches = [r for r in records if r.get('case_id') == case and
               (r.get('mode') == 'memory_only' if memory_only else
                r.get('mode') != 'memory_only' and r.get('turn') == turn)]
    if len(matches) != 1:
        raise ValueError('Trace must identify exactly one turn or probe')
    return matches[0]


def owner_bound_query(message):
    """Diagnostic deictic binding over parsed user-memory spans, not free text.

    Preserve controls, punctuation, declarations, quotes and other tasks. This
    does not run on production user messages or change the saved source text.
    """
    from knowledge.task_plan import plan_turn_tasks

    result = message
    for task in reversed(plan_turn_tasks(message)):
        if task.kind != 'memory':
            continue
        bound = re.sub(r'我(?=的|来自|叫|(?:目前|现在)?住)', '当前对话用户', task.original, count=1)
        result = result[:task.start] + bound + result[task.end:]
    return result


def owner_bound_messages(messages, trace):
    from html import escape

    original = trace['message']
    bound = owner_bound_query(original)
    if bound == original:
        raise ValueError('No closed personal memory span to bind')
    terminal = '<user_query>\n' + escape(original, quote=False) + '\n</user_query>'
    if not messages or messages[-1]['role'] != 'user' or not messages[-1]['content'].endswith(terminal):
        raise ValueError('Expected unchanged complete terminal user query')
    result = [dict(m) for m in messages]
    result[-1]['content'] = (result[-1]['content'][:-len(terminal)] +
                             '<user_query>\n' + escape(bound, quote=False) + '\n</user_query>')
    return result


def separated_query_messages(messages, trace):
    """Diagnostic native-message boundary; no new instructions or owner rewrite."""
    from html import escape

    query = trace['message']
    terminal = '<user_query>\n' + escape(query, quote=False) + '\n</user_query>'
    if not messages or messages[-1]['role'] != 'user' or not messages[-1]['content'].endswith(terminal):
        raise ValueError('Expected exact terminal user query')
    reference = messages[-1]['content'][:-len(terminal)]
    if not reference.strip():
        raise ValueError('No reference message to separate')
    return [*(dict(m) for m in messages[:-1]), {'role': 'user', 'content': reference},
            {'role': 'user', 'content': query}]


def persona_variants(messages):
    """Diagnostic system ablation; never a proposed production persona removal."""
    if not messages or messages[0]['role'] != 'system':
        raise ValueError('Expected a system message')
    system = messages[0]['content']
    directive = re.findall(r'^- 始终以第一人称和[^\n]+身份自然交流。\n', system, flags=re.M)
    marker = '【当前关系】'
    if len(directive) != 1 or not system.startswith('【人物身份】') or system.count(marker) != 1:
        raise ValueError('Expected one complete structured persona and first-person directive')
    without_directive = [dict(m) for m in messages]
    without_directive[0]['content'] = system.replace(directive[0], '', 1)
    without_profile = [dict(m) for m in messages]
    without_profile[0]['content'] = system[system.index(marker):]
    return {'original': messages, 'without_first_person_directive': without_directive,
            'without_profile': without_profile}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--case', required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--turn', type=int)
    target.add_argument('--memory-only-probe', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--selection-audit', type=Path)
    parser.add_argument('--evidence-view', choices=['original', 'card_evidence', 'annotated', 'speaker_compare',
                                                 'background_compare', 'task_compare', 'selection_compare',
                                                 'owner_compare', 'message_compare', 'persona_compare'], default='original')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError('repeats must be 1..10')
    matches = [json.loads(line) for line in args.trace.read_text(encoding='utf-8').splitlines() if line.strip()]
    matches = [select_trace(matches, args.case, turn=args.turn, memory_only=args.memory_only_probe)]
    call = matches[0]['model_calls'][0]
    if call['parameters'].get('lora_name') not in (None, '', 'default'):
        raise ValueError('No-LoRA inputs required')
    messages = call['messages']
    if args.evidence_view == 'card_evidence':
        messages = card_evidence_messages(messages, matches[0]['generation']['plan']['retrieval']['documents'])
    elif args.evidence_view == 'annotated':
        messages = annotated_evidence_messages(messages, matches[0]['generation']['plan']['retrieval'])
    selection_hash = None
    if args.evidence_view == 'persona_compare':
        alternatives = persona_variants(messages)
    elif args.evidence_view == 'message_compare':
        alternatives = {'original': messages, 'separate_user_query': separated_query_messages(messages, matches[0])}
    elif args.evidence_view == 'owner_compare':
        alternatives = {'original': messages, 'bound_user_reference': owner_bound_messages(messages, matches[0])}
    elif args.evidence_view == 'selection_compare':
        if args.selection_audit is None:
            raise ValueError('selection_compare requires --selection-audit')
        audits = [json.loads(line) for line in args.selection_audit.read_text(encoding='utf-8').splitlines() if line.strip()]
        audits = [r for r in audits if r.get('query') == matches[0]['message']]
        if len(audits) != 1:
            raise ValueError('Expected exactly one same-query retrieval audit')
        selection_hash = hashlib.sha256(args.selection_audit.read_bytes()).hexdigest()
        alternatives = {'original': messages, 'selection_coverage': selection_evidence_messages(messages, matches[0], audits[0])}
    elif args.evidence_view == 'speaker_compare':
        alternatives = {'original': messages, 'typed_source_lines': speaker_evidence_messages(
            messages, matches[0]['generation']['plan']['retrieval'])}
    elif args.evidence_view == 'background_compare':
        alternatives = {'original': messages, 'without_background': without_background_messages(
            messages, matches[0]['generation']['plan']['retrieval'])}
    elif args.evidence_view == 'task_compare':
        alternatives = {'residual_query': messages, 'full_query_same_history': full_task_messages(messages, matches[0])}
    else:
        alternatives = variants(messages)
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    for repeat in range(args.repeats):
        order = list(alternatives)
        if repeat % 2:
            order.reverse()
        for mode in order:
            messages = alternatives[mode]
            params, assumed = replay_parameters(call, matches[0]['generation']['plan']['generation'])
            body = dict(model=args.model, messages=messages, stream=False, seed=repeat,
                        temperature=params['temperature'], max_tokens=params['max_tokens'],
                        top_p=params['top_p'],
                        repetition_penalty=params['repetition_penalty'],
                        frequency_penalty=params['frequency_penalty'],
                        chat_template_kwargs={'enable_thinking': params['enable_thinking']})
            headers = {'Content-Type': 'application/json'}
            if os.getenv('VLLM_API_KEY'):
                headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
            request = Request(args.endpoint, data=json.dumps(body).encode(), headers=headers)
            started = time.monotonic()
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            record = dict(case_id=args.case, turn=args.turn, mode=mode, seed=repeat,
                          source_mode='memory_only' if args.memory_only_probe else 'ordinary',
                          evidence_view=args.evidence_view,
                          source_sha256=source_hash, request=body, response=result,
                          selection_sha256=selection_hash,
                          parameters_from_plan_not_observed=assumed,
                          seconds=time.monotonic() - started,
                          assessment='diagnostic_only_not_a_quality_pass')
            with (args.output / 'replays.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps(dict(mode=mode, seed=repeat, seconds=record['seconds'])), flush=True)


if __name__ == '__main__':
    main()
