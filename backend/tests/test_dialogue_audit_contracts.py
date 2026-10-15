"""Transfer tests for dialogue scope and evaluation infrastructure."""
import pytest
from evaluation.dialogue_audit import load_cases

from db.database import SQLiteDB
from knowledge.dialogue_query import contextual_retrieval_query
from knowledge.retrieval_core.registry import KnowledgeDomainConfig


def test_multiline_suite_and_incomplete_records(tmp_path):
    path = tmp_path / 'suite.txt'
    path.write_text('## 01 日常\n多行|空历史|第一行\n第二行|检查|再聊|检查|结束|检查\n', encoding='utf-8')
    case = load_cases(path)[0]
    assert case['turns'][0]['message'] == '第一行\n第二行'
    assert load_cases(path)[0]['id'] == case['id']
    path.write_text('## 01 日常\n未完成|空历史|一句', encoding='utf-8')
    with pytest.raises(ValueError, match='Incomplete'):
        load_cases(path)


def add(db, text, character=None, kind='private', sender='same-user'):
    return db.add_message(dict(sessionId='same-room', sessionType=kind,
        platform='web', adapter='web-character', senderId=sender,
        characterId=character, message=text, reply='回复', createdAt='2026-01-01T00:00:00'))


@pytest.mark.parametrize('character', ['role-a', 'role-b'])
def test_character_history_filters_before_limit_and_excludes_legacy(tmp_path, character):
    db = SQLiteDB(tmp_path / 'history.db')
    add(db, '应保留', character)
    for _ in range(55):
        add(db, '别的人物', 'unrelated')
    add(db, '未标记旧记录')
    add(db, '别的用户', character, sender='different')
    history = db.list_conversation_history('web', 'web-character', 'same-user', 'private', 'same-room',
                                           limit=1, character_id=character)
    assert [x['content'] for x in history] == ['应保留', '回复']


def test_history_type_collision_and_equal_timestamp_order(tmp_path):
    db = SQLiteDB(tmp_path / 'history.db')
    add(db, '群聊第一轮', kind='group')
    add(db, '群聊第二轮', kind='group')
    add(db, '同号频道', kind='channel')
    history = db.list_conversation_history('web', 'web-character', 'same-user', 'group', 'same-room')
    assert [x['content'] for x in history if x['role'] == 'user'] == ['群聊第一轮', '群聊第二轮']


@pytest.mark.parametrize('name', ['测试人物甲', '另一作品人物乙'])
def test_retrieval_followup_inherits_only_user_topic(tmp_path, name):
    configs = [KnowledgeDomainConfig('test', tmp_path, lambda _: [], aliases={name: name})]
    history = [{'role': 'user', 'content': f'{name}是什么身份？'},
               {'role': 'assistant', 'content': '我猜是某国总统，这个猜测不可作为查询事实。'}]
    query = contextual_retrieval_query('请给出原文依据。', history, configs)
    assert name in query and '总统' not in query
    assert contextual_retrieval_query('换个话题，我今天看书了。', history, configs) == '换个话题，我今天看书了。'
    history += [{'role': 'user', 'content': '我今天看书了。'}]
    assert contextual_retrieval_query('那有什么特点？', history, configs) == '那有什么特点？'


@pytest.mark.parametrize('query', [
    '能指出支持你说法的原作内容吗？',
    '可以列出相关的原著段落吗？',
    '这个结论出自哪一章节？',
    '请提供引文。',
    '出处？',
])
def test_source_task_followup_routes_to_authorized_user_topic(tmp_path, query):
    configs = [KnowledgeDomainConfig('test', tmp_path, lambda _: [], aliases={'测试主角': '测试主角'})]
    history = [{'role': 'user', 'content': '测试主角和朋友是什么关系？'},
               {'role': 'assistant', 'content': '某个未验证的猜测'},
               {'role': 'user', 'content': '这种关系有没有冲突？'}]
    resolved = contextual_retrieval_query(query, history, configs)
    assert '测试主角' in resolved and query in resolved
    assert '未验证' not in resolved
    assert contextual_retrieval_query(query, [], configs) == query
    history += [{'role': 'user', 'content': '我今天读完了另一本原著。'}]
    assert contextual_retrieval_query(query, history, configs) == query


def test_source_mention_without_request_does_not_inherit_topic(tmp_path):
    configs = [KnowledgeDomainConfig('test', tmp_path, lambda _: [], aliases={'测试主角': '测试主角'})]
    history = [{'role': 'user', 'content': '测试主角是什么身份？'}]
    query = '我今天读完了另一本原著。'
    assert contextual_retrieval_query(query, history, configs) == query


@pytest.mark.parametrize('reply', ['如果你喜欢某种食物，可以再讨论。', '你喜欢独处还是结伴呢？', '你习惯早起吗？'])
def test_guard_does_not_convert_questions_or_conditions_to_claims(reply):
    from character.output_guard import ReplyGuard, validate_reply
    assert 'unsupported_user_fact' not in validate_reply(reply, ReplyGuard(forbid_unsupported_user_fact=True))


@pytest.mark.asyncio
async def test_continuity_is_not_a_persistent_memory_match(tmp_path):
    from services.character_context import TurnInput, build_character_context_service
    db = SQLiteDB(tmp_path / 'memory.db')
    svc = build_character_context_service(db)
    prepared = await svc.prepare_turn(TurnInput(message='我有什么偏好？', platform='web', adapter='audit',
        sender_id='user', conversation_id='user', conversation_type='private',
        history=({'role': 'user', 'content': '你好'},)), 'tsukiyashiro_kisaki')
    # Conversation cues have their own channel; their presence is not a fact
    # memory hit and must not activate the persistent-memory attribution policy.
    assert '你好' in prepared.compiled.conversation_reference_context
    assert not prepared.compiled.reference_context
    assert prepared.compiled.memory_status == 'no_match'


@pytest.mark.asyncio
async def test_grounded_names_do_not_trigger_unprompted_lore_retry():
    from unittest.mock import AsyncMock

    from character.output_guard import ReplyGuard
    from inference.generation_request import GenerationRequest, RetrievalResult, generate_character_response
    model = AsyncMock(return_value='人物甲与人物乙是朋友。')
    result = await generate_character_response(GenerationRequest(message='他们有什么关系？',
        retrieval=RetrievalResult(status='ok', evidence='人物甲与人物乙是朋友。'),
        reply_guard=ReplyGuard(forbidden_terms=('人物甲', '人物乙', '人物丙'))), model)
    model.assert_awaited_once()
    assert not result.guard_retried


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['private', 'group', 'channel'])
async def test_postgres_history_contract_uses_bound_scope_and_stable_order(kind, monkeypatch):
    # Exercise the real PG query builder without connecting to production.
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    pytest.importorskip('asyncpg')
    monkeypatch.setenv('DATABASE_URL', 'postgresql+asyncpg://test:test@localhost/test')
    from db.pg_database import PgDatabase
    session = MagicMock()
    session.execute = AsyncMock(return_value=SimpleNamespace(fetchall=lambda: []))
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=session)
    manager.__aexit__ = AsyncMock(return_value=False)
    fake = SimpleNamespace(async_session=lambda: manager)
    assert await PgDatabase.list_conversation_history(fake, 'web', 'audit', 'user', kind, 'room',
                                                      character_id='role') == []
    stmt, params = session.execute.call_args.args
    assert '"characterId" = :character_id' in str(stmt)
    assert '"createdAt" DESC, id DESC' in str(stmt)
    assert params['character_id'] == 'role'
    if kind != 'private':
        assert params['conversation_type'] == kind


def test_scope_migration_preserves_legacy_rows_and_is_idempotent(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    path = Path(__file__).resolve().parents[1] / 'alembic/versions/010_message_character_scope.py'
    spec = importlib.util.spec_from_file_location('scope_migration', path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine(f'sqlite:///{tmp_path / "legacy.db"}')
    with engine.begin() as conn:
        conn.execute(sa.text('CREATE TABLE messages (id INTEGER PRIMARY KEY, message TEXT)'))
        conn.execute(sa.text("INSERT INTO messages VALUES (1, 'legacy')"))
        monkeypatch.setattr(migration, 'op', Operations(MigrationContext.configure(conn)))
        migration.upgrade()
        migration.upgrade()
        assert conn.execute(sa.text('SELECT message, characterId FROM messages')).one() == ('legacy', None)
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [40, 120])
async def test_persistent_recall_without_chat_history_after_many_newer_records(tmp_path, count, monkeypatch):
    from services.character_context import TurnInput, build_character_context_service
    monkeypatch.setenv('CAHM_SEMANTIC_MEMORY_ENABLED', 'false')
    db = SQLiteDB(tmp_path / 'persistent.db')
    svc = build_character_context_service(db)
    from datetime import datetime, timezone

    from repositories.character_memory import DatabaseCharacterMemoryRepository

    from character.models import MemoryItem, UserScope

    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope('web', 'audit', 'u1', 'room', 'group')
    observed = datetime.now(timezone.utc)
    assert await repo.capture_source('tsukiyashiro_kisaki', scope,
        source_message_id='source-1', body='我叫林青。', observed_at=observed) == 'recorded'
    await repo.append_claim('tsukiyashiro_kisaki', scope, MemoryItem('', 'user_fact', '用户姓名是林青', .95),
        memory_key='user_name', evidence=('我叫林青。',), source_message_id='source-1',
        observed_at=observed.isoformat(), confidence=.99, metadata={'origin': 'rule_v2'})
    for index in range(count):
        db.add_or_update_character_memory(character_id='tsukiyashiro_kisaki', platform='web',
            adapter='audit', sender_id='u1', conversation_type='group', conversation_id='room',
            memory_type='shared_event', memory_key=f'misc-{index}', content=f'整理物品编号{index}')
    recalled = await svc.prepare_turn(TurnInput('我叫什么名字？', 'web', 'audit', 'u1', 'room', 'group'),
                                      'tsukiyashiro_kisaki')
    assert not recalled.history
    assert '林青' in recalled.compiled.reference_context
    other = await svc.prepare_turn(TurnInput('我叫什么名字？', 'web', 'audit', 'u2', 'room', 'group'),
                                   'tsukiyashiro_kisaki')
    assert '林青' not in other.compiled.reference_context
    assert other.compiled.memory_status == 'no_match'


@pytest.mark.parametrize('query', [
    '请给出原文', '能指出支持你说法的原作内容吗？',
    '请提供相关的原著段落。', '出处？', '能列出引文吗？',
])
def test_rag_deduplicates_scene_and_delivers_exact_excerpt(monkeypatch, query, tmp_path):
    from types import SimpleNamespace

    from knowledge.multiscale_rag import service
    from knowledge.multiscale_rag.source_text import OriginalTextExtractor
    from knowledge.retrieval_core.documents import SourceReference
    source = SourceReference('test.txt', 4, 6)
    (tmp_path / 'test.txt').write_text('占位一\n占位二\n占位三\n未被改写的原文。\n完整条件：仅离线。\n最终例外：仍需审批。', encoding='utf-8')
    docs = [SimpleNamespace(id=f'fact-{i}', document_type='fact', title=f'事实{i}',
                            summary=f'摘要{i}', metadata={'scene_id': 'shared-scene'}, source=source,
                            domain_id='test', content=f'摘要{i}', reality_status='objective',
                            temporal_scope='current', content_scope='main_story')
            for i in range(2)]
    candidates = [SimpleNamespace(document=d, to_dict=lambda: {}) for d in docs]
    instance = object.__new__(service.RoutedMultiScaleService)
    instance.identity_coverage = False
    instance.config = SimpleNamespace(domain_id='test')
    instance.analyzer = None
    instance.indexes = {}
    instance.retrievers = {service.CARD_TYPES: SimpleNamespace(search=lambda *a, **kw: candidates)}
    instance.reranker = None
    instance.by_id = {'shared-scene': SimpleNamespace(id='shared-scene', title='共同场景', summary='背景内容')}
    instance.evidence_by_parent = {'fact-0': SimpleNamespace(id='evidence-0', source=source)}
    instance.extractor = OriginalTextExtractor(tmp_path)
    monkeypatch.setattr(service, 'analyze_explicit_domain', lambda *a: SimpleNamespace(entities=[]))
    monkeypatch.setattr(service, 'choose_card_types', lambda *a: service.CARD_TYPES)
    monkeypatch.setattr(service, 'rerank_with_title_frames', lambda *a, **kw: candidates)
    result = instance.retrieve(query)
    assert result['context_text'].count('【父场景】') == 1
    assert result['context_text'].startswith('【原文摘录】test.txt L4-6\n未被改写的原文。')
    assert len(result['citations']) == 2
    assert result['raw_source_status'] == 'available'
    # The same request must not relabel a summary as original source text.
    instance.extractor = None
    with pytest.raises(RuntimeError, match='source extractor'):
        instance.retrieve(query)
    instance.extractor = OriginalTextExtractor(tmp_path)
    (tmp_path / 'test.txt').unlink()
    with pytest.raises(FileNotFoundError):
        instance.retrieve(query)


@pytest.mark.parametrize('question', ['我还喜欢绘画吗？', '我已经完成申请了吗？', '我拿到录取了吗？', '我喜欢哪种运动？'])
def test_interrogative_is_not_a_high_confidence_celebration(question):
    from character.situation_analyzer import SituationAnalyzer
    state = SituationAnalyzer().estimate(question)
    acts = {a.signal_id: a.score for a in state.user_acts}
    assert acts.get('positive_sharing', 0) < .5
    assert acts.get('information_request', 0) >= .5


def test_real_celebration_and_question_can_coexist():
    from character.situation_analyzer import SituationAnalyzer
    state = SituationAnalyzer().estimate('我的申请通过了！接下来需要什么材料？')
    acts = {a.signal_id: a.score for a in state.user_acts}
    assert acts.get('positive_sharing', 0) >= .5
    assert acts.get('information_request', 0) >= .5
