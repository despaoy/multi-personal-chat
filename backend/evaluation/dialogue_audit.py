"""Isolated no-LoRA dialogue audit through the production orchestration path.

Records evidence, not a model-generated similarity score. Operator directives
are never sent as user text; unsupported fixtures remain explicit coverage gaps.
Run in an isolated process; instrumentation is never installed in the web app.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path


def load_history_fixtures(path: Path | None, cases: list[dict]) -> dict:
    """Optional observed/synthetic client histories, isolated to the first turn."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    known = {case['id'] for case in cases}
    if not isinstance(data, dict) or set(data) - known:
        raise ValueError('History fixtures must map known case IDs to messages')
    for messages in data.values():
        if not isinstance(messages, list) or not messages:
            raise ValueError('History fixture must contain messages')
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                    or not isinstance(message['role'], str)
                    or message['role'] not in {'user', 'assistant'}
                    or not isinstance(message['content'], str) or not message['content'].strip()):
                raise ValueError('Only nonempty user/assistant fixture messages are allowed')
    return data


def load_trajectories(path: Path) -> list[dict]:
    """Variable-length user-only journeys; session changes never change owner.

    Rubrics remain observer data. No arbitrary request overrides, injected
    assistant history, database operations or setup instructions are executed.
    """
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, list) or not data:
        raise ValueError('Trajectories must be a nonempty list')
    cases = []
    seen = set()
    for row in data:
        if (not isinstance(row, dict) or set(row) != {'title', 'turns'}
                or not isinstance(row['title'], str) or not row['title'].strip()
                or not isinstance(row['turns'], list) or not 1 <= len(row['turns']) <= 200):
            raise ValueError('Expected title and 1..200 user turns')
        category = '10 longitudinal_memory'
        case_id = hashlib.sha256((category + '/' + row['title']).encode()).hexdigest()[:12]
        if case_id in seen:
            raise ValueError('Duplicate trajectory title')
        seen.add(case_id)
        for turn in row['turns']:
            if (not isinstance(turn, dict) or set(turn) != {'message', 'rubric', 'session'}
                    or not isinstance(turn['message'], str) or not 1 <= len(turn['message'].strip()) <= 8000
                    or not isinstance(turn['rubric'], str)
                    or not isinstance(turn['session'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', turn['session'])):
                raise ValueError('Expected message, rubric and bounded session label')
        cases.append(dict(id=case_id, category=category, title=row['title'], setup='isolated_user_journey',
                          split='holdout' if int(case_id, 16) % 4 == 0 else 'dev', skip_reason='', turns=row['turns']))
    return cases


def load_cases(path: Path) -> list[dict]:
    if path.suffix.lower() == '.json':
        return load_trajectories(path)
    cases = []
    category = ''
    pending = ''
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if pending:
            if line.startswith('## '):
                raise ValueError(f'Incomplete record: {pending[:50]}')
            line = pending + '\n' + line
            pending = ''
        if line.startswith('## '):
            category = line[3:]
        if not line or line.startswith('#'):
            continue
        if line.count('|') < 7:
            pending = line
            continue
        fields = line.split('|')
        if len(fields) != 8:
            raise ValueError(f'Expected 8 fields: {line[:50]}')
        title, setup, *turns = fields
        case_id = hashlib.sha256((category + '/' + title).encode()).hexdigest()[:12]
        scope = int(category.split()[0])
        reason = ''
        if scope not in {*range(1, 13), 15}:
            reason = 'requires_external_fixture_or_out_of_scope'
        elif any('【' in text for text in turns[::2]):
            reason = 'requires_operator_fixture'
        elif re.search(r'夹具|预置|固定|T0|测试时钟|手动|管理员|开启|关闭|注入|构造|插入|临界|已存|已有|close|两请求|A和|F-RAG|F-KB|扩到|超过\d', setup):
            reason = 'requires_state_fixture'
        cases.append(dict(id=case_id, category=category, title=title, setup=setup,
                          split='holdout' if int(case_id, 16) % 4 == 0 else 'dev',
                          skip_reason=reason,
                          turns=[dict(message=turns[i], rubric=turns[i+1]) for i in range(0, 6, 2)]))
    if pending:
        raise ValueError(f'Incomplete record: {pending[:50]}')
    # Actual production-domain questions: no hand-crafted answer fed to retrieval.
    rag_cases = [
        ('identity', ['月社妃是谁？', '她和琉璃是什么关系？', '谁是哥哥？']),
        ('book', ['月社妃是哪本魔法之书的主人公？', '确定不是《蓝宝石的存在证明》吗？', '请给出原文依据。']),
        ('unknown', ['月社妃银行卡的密码是什么？', '不知道就直说，不要猜。', '那她和彼方是什么关系？']),
        ('followup', ['说说月社妃和夜子的关系。', '这种关系中有没有冲突？', '能指出支持你说法的原作内容吗？']),
        ('unsupported', ['月社妃最喜欢的手机品牌是哪家？', '我猜是苹果，你觉得呢？', '只告诉我能确定的部分。']),
        ('daily_switch', ['月社妃和理央是什么关系？', '换个话题，我今天刚整理好书架。', '我只是随便分享一下，不用讲剧情。']),
        ('attribution', ['我现在住在长沙。', '月社妃住在长沙吗？', '请分清我的资料和角色的原作设定。']),
        ('false_premise', ['琉璃是月社妃的弟弟，对吗？', '请以原作为准，不用顺着我。', '那准确的关系是什么？']),
    ]
    for i, (name, turns) in enumerate(rag_cases):
        cases.append(dict(id='rag-' + name, category='13 production_RAG', title=name, setup='existing_index',
                          split='holdout' if i in {3, 7} else 'dev', skip_reason='',
                          turns=[dict(message=m, rubric='核对证据、引用、未知信息与主体；不凭空补造原作事实') for m in turns]))
    return cases


async def run(args, cases: list[dict], out: Path):
    # Imports must follow DATABASE_PATH initialization in main().
    import api.generate as gen
    from db.adapter import db
    from db.schemas import MessageRequest
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from services.character_context import build_character_context_service

    service = build_character_context_service(db)
    repo = DatabaseCharacterMemoryRepository(db)
    current = {}
    prepared_turn = None
    original_generation = gen.generate_character_response

    async def traced_generation(request, generate):
        current['generation_features'] = {'independent_tasks_enabled': request.independent_tasks_enabled}
        calls = []
        async def traced_model(**kwargs):
            if kwargs.get('lora_name') not in {None, '', 'default'}:
                raise RuntimeError('No-LoRA audit received an adapter')
            t = time.monotonic()
            reply = await generate(**kwargs)
            calls.append(dict(messages=kwargs['messages'], reply=reply, seconds=time.monotonic()-t,
                              parameters={k: kwargs.get(k) for k in (
                                  'lora_name', 'temperature', 'max_tokens', 'top_p', 'enable_thinking',
                                  'repetition_penalty', 'frequency_penalty')}))
            return reply
        result = await original_generation(request, traced_model)
        current['generation'] = asdict(result)
        current['model_calls'] = calls
        return result
    gen.generate_character_response = traced_generation

    class TracedService:
        def __init__(self, inner=service):
            self.inner = inner

        async def prepare_turn(self, turn, character_id):
            nonlocal prepared_turn
            t = time.monotonic()
            prepared = await self.inner.prepare_turn(turn, character_id)
            current['prepared'] = asdict(prepared)
            prepared_turn = prepared
            current['prepare_seconds'] = time.monotonic()-t
            return prepared

        async def complete_turn(self, prepared, turn, reply, **kwargs):
            started = time.monotonic()
            outcome = await self.inner.complete_turn(prepared, turn, reply, **kwargs)
            current['write_seconds'] = time.monotonic() - started
            current['write_outcome'] = asdict(outcome)
            return outcome

    chat = gen._build_chat_generation_service(gen.inference_runtime, TracedService(), db)
    selected = [c for c in cases if not c['skip_reason'] and c['split'] == args.split]
    if args.limit:
        selected = selected[:args.limit]
    meta = dict(split=args.split, selected=len(selected), total_inventory=len(cases),
                unsupported=sum(bool(c['skip_reason']) for c in cases), lora=None,
                model=os.getenv('VLLM_SERVED_MODEL_NAME'),
                independent_tasks_enabled=os.getenv('CHARACTER_INDEPENDENT_TASKS_ENABLED', 'false').lower() == 'true',
                switches={k: os.getenv(k, 'default') for k in ['MEMORY_LLM_ENABLED', 'CAHM_SEMANTIC_MEMORY_ENABLED',
                    'CONTEXTUAL_MEMORY_SELECTION_ENABLED', 'CONTEXTUAL_DECISION_POLICY_ENABLED', 'VLLM_MAX_MODEL_LEN']})
    (out / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    for case in selected:
        for index, turn in enumerate(case['turns']):
            current.clear()
            prepared_turn = None
            current.update(case_id=case['id'], category=case['category'], title=case['title'],
                           turn=index+1, message=turn['message'], rubric=turn['rubric'])
            session_id = case['id'] + ('-' + turn['session'] if 'session' in turn else '')
            current['session_id'] = session_id
            request = MessageRequest(message=turn['message'], characterId='tsukiyashiro_kisaki', loraName='default',
                platform='web', adapter='dialogue-audit', senderId=case['id'], userId=case['id'],
                sessionId=session_id, conversationId=session_id, sourceMessageId=f'{case["id"]}-{index+1}')
            initial_history = getattr(args, 'history_fixtures', {}).get(case['id']) if index == 0 else None
            if initial_history:
                request = request.model_copy(update={'history': initial_history})
                current['initial_history_fixture'] = initial_history
            start = time.monotonic()
            try:
                response = await chat.generate(request)
                current['response'] = response.model_dump()
                current['status'] = 'generated_not_yet_reviewed'
            except Exception as exc:
                current.update(status='error', error_type=type(exc).__name__, error=str(exc))
            current['seconds'] = time.monotonic()-start
            # Observer reads must not consume the production write deadline.
            if prepared_turn is not None:
                try:
                    current['records_after'] = await repo.list_memory_records(
                        prepared_turn.character_id, prepared_turn.user_scope, limit=None, include_inactive=True)
                    items, candidates = await service._memory_service.load_relevant_memories(
                        prepared_turn.character_id, prepared_turn.user_scope, turn['message'])
                    current['memory_only_recall'] = dict(items=[asdict(x) for x in items], candidates=candidates)
                except Exception as exc:
                    current['observer_error'] = f'{type(exc).__name__}: {exc}'
            with (out / 'traces.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(current, ensure_ascii=False, default=str) + '\n')
            summary = {k: current[k] for k in ('case_id', 'category', 'title', 'turn', 'message', 'rubric', 'status')}
            summary.update(reply=current.get('response', {}).get('reply'),
                           calls=len(current.get('model_calls', [])),
                           writes=current.get('write_outcome'),
                           memory=current.get('prepared', {}).get('compiled', {}).get('reference_context'),
                           strategies=current.get('prepared', {}).get('decision', {}).get('strategy_ids'),
                           error=current.get('error'))
            with (out / 'review.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(summary, ensure_ascii=False, default=str) + '\n')
            print(json.dumps(dict(case=case['id'], turn=index+1, status=current['status'], seconds=round(current['seconds'],2))), flush=True)
        if args.memory_only_probe and case['category'].split()[0] in {'05', '06', '07', '09', '10'}:
            class EmptyHistory:
                async def list_recent_conversation_history(self, *args, **kwargs):
                    return []
            isolated_service = copy.copy(service)
            isolated_service._message_repo = EmptyHistory()
            probe_chat = gen._build_chat_generation_service(gen.inference_runtime, TracedService(isolated_service), db)
            current.clear()
            current.update(case_id=case['id'], title=case['title'], message=request.message, mode='memory_only')
            try:
                response = await probe_chat.generate(request.model_copy(update={'history': []}),
                                                     persist_message=False, record_invocation=False)
                current.update(response=response.model_dump(), status='generated_not_yet_reviewed')
            except Exception as exc:
                current.update(status='error', error_type=type(exc).__name__, error=str(exc))
            with (out / 'probes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(current, ensure_ascii=False, default=str) + '\n')
    print('AUDIT_OUTPUT=' + str(out), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'holdout'], default='dev')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--memory-only-probe', action='store_true')
    parser.add_argument('--history-fixtures', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = load_cases(args.suite)
    args.history_fixtures = load_history_fixtures(args.history_fixtures, cases)
    (args.output / 'history-fixtures.json').write_text(
        json.dumps(args.history_fixtures, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'inventory.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'suite.sha256').write_text(hashlib.sha256(args.suite.read_bytes()).hexdigest(), encoding='utf-8')
    (args.output / 'suite.txt').write_bytes(args.suite.read_bytes())
    backend = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(backend)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(backend.rglob('*.py')) if '__pycache__' not in p.parts}
    (args.output / 'source-hashes.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    if args.prepare_only:
        from collections import Counter
        print(json.dumps(dict(total=len(cases), eligible=dict(Counter(c['split'] for c in cases if not c['skip_reason'])),
                              skipped=dict(Counter(c['skip_reason'] for c in cases if c['skip_reason']))), ensure_ascii=False))
        return
    os.environ['USE_POSTGRESQL'] = 'false'
    os.environ['DATABASE_PATH'] = str(args.output / 'isolated.sqlite')
    os.environ['ENVIRONMENT'] = 'development'
    asyncio.run(run(args, cases, args.output))


if __name__ == '__main__':
    main()
