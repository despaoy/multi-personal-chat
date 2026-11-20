import asyncio
import re
from functools import partial

import httpx
import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from inference.generation_request import GenerationRequest, generate_character_response
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository
from services.character_chat import CharacterChatService, prepare_request
from services.character_chat_evidence import CharacterChatCompletion, evidence_snapshot
from services.character_chat_vectors import embedding_identity
from services.character_source_retrieval import retrieve_source_context


class Provider:
    model_id = 'test'
    model_fingerprint = 'test-weights'
    dimension = 3

    def embed_query(self, message):
        return [1, 0, 0]


def test_unfinished_index_allows_reply_without_source_admission(tmp_path):
    db = SQLiteDB(tmp_path / 'pending-source-chat.db')
    sources = CharacterSourceRepository(db)
    chats = CharacterChatRepository(db)
    try:
        char = sources.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        corpus = sources.publish('owner', char['id'], 1, [{'id': 'private', 'title': '资料', 'content': '待索引正文'}])
        session = chats.create('owner', char['id'], '普通聊天')
        prompt = prepare_request(session, [], "她当时见到的那个人是谁？").persona_prompt
        assert "只问一个聚焦的澄清问题" in prompt
        assert "不要先复述背景、补猜情节或添加引用" in prompt

        class Index:
            index = 'test-index'
            dimension = 3

            async def candidates(self, *args, **kwargs):
                pytest.fail('Unfinished corpus must not use search candidates')

        request = GenerationRequest(message='想聊聊天')
        result = asyncio.run(retrieve_source_context(request, session, database=db, index=Index(), provider=Provider()))
        assert result.retrieval.status == 'character_abstention'
        assert result.retrieval.reason == 'retrieval_unavailable'
        assert not result.retrieval.citations and not result.retrieval.evidence
        sources.revoke('owner', corpus['id'])
        with pytest.raises(CharacterNotFoundError):
            asyncio.run(retrieve_source_context(request, session, database=db, index=Index(), provider=Provider()))
    finally:
        db.close_connection()


@pytest.mark.parametrize('revoke_during_reply', [False, True])
def test_source_lookup_reaches_prompt_and_commits_only_live_authority(tmp_path, revoke_during_reply):
    db = SQLiteDB(tmp_path / 'source-chat.db')
    sources = CharacterSourceRepository(db)
    chats = CharacterChatRepository(db)
    try:
        char = sources.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        corpus = sources.publish('owner', char['id'], 1, [{'id': 'chapter1', 'title': '第一章', 'content': '小满在傍晚把信送到了灯塔。'}])
        session = chats.create('owner', char['id'], '问原作')
        candidate = sources.chunks('owner', corpus['id'])[0]
        identity = embedding_identity(Provider())
        sources.mark_indexed('owner', corpus['id'], {'index_name': 'test-index', 'embedding_identity': identity,
                                                     'chunk_count': 1, 'corpus_digest': corpus['content_digest']})

        class Index:
            index = 'test-index'
            dimension = 3

            async def candidates(self, scope, query, vector, *, limit):
                assert scope.owner_id == 'owner' and scope.corpus_revision == corpus['id']
                assert limit > 8
                return [{**candidate, 'embedding_identity': identity, 'content': '索引里的伪造原文'}]

        async def complete(request, _session, _request_id):
            async def generate(**kwargs):
                prompt = str(kwargs)
                assert '小满在傍晚把信送到了灯塔。' in prompt
                assert '索引里的伪造原文' not in prompt
                if revoke_during_reply:
                    sources.revoke('owner', corpus['id'])
                namespace = re.search(r'\[\[cite:([0-9a-f]{12}):S1\]\]', prompt).group(1)
                return '小满在傍晚把信送到了灯塔。' + f'[[cite:{namespace}:S1]]'
            result = await generate_character_response(request, generate)
            return CharacterChatCompletion(result.reply, 'controlled', evidence_snapshot(result))

        service = CharacterChatService(db, complete, sources=partial(retrieve_source_context, database=db, index=Index(), provider=Provider()))
        call = service.generate('owner', session['id'], revision=1, request_id='source-test', message='原作里谁把信送到了灯塔？')
        if revoke_during_reply:
            with pytest.raises(CharacterNotFoundError):
                asyncio.run(call)
            assert chats.turns('owner', session['id']) == []
        else:
            turn = asyncio.run(call)
            assert turn['evidence']['sources'][0]['title'] == '第一章'
            assert turn['evidence']['sources'][0]['excerpt'] == candidate['content']
            updated = chats.get('owner', session['id'])
            sources.revoke('owner', corpus['id'])
            with pytest.raises(RevisionConflictError):
                chats.commit_turn(updated, 'late', '追问', '迟到的答案', 'test', source_corpus_id=corpus['id'])
            assert len(chats.turns('owner', session['id'])) == 1
    finally:
        db.close_connection()


@pytest.mark.parametrize("mode", ["literal", "model", "model_unavailable", "invalid_order", "search_unavailable"])
def test_reranking_expands_only_authoritative_neighbor_lines(tmp_path, mode):
    db = SQLiteDB(tmp_path / 'source-ranking.db')
    sources = CharacterSourceRepository(db)
    chats = CharacterChatRepository(db)
    try:
        char = sources.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        corpus = sources.publish('owner', char['id'], 1, [
            {'id': 'irrelevant', 'title': '车站', 'content': '旅人还未出发。'},
            {'id': 'original', 'title': '灯塔', 'content': '另一段落。\n\n风声渐小。\n小满送信到灯塔。\n门卫让她进去。\n\n不应跨段。'},
        ])
        session = chats.create('owner', char['id'], '问原作')
        chunks = sources.chunks('owner', corpus['id'], max_chars=14)
        anchor = next(c for c in chunks if c['source_id'] == 'original' and c['line_start'] == 4)
        unrelated = next(c for c in chunks if c['source_id'] == 'irrelevant')
        identity = embedding_identity(Provider())
        sources.mark_indexed('owner', corpus['id'], {'index_name': 'test-index', 'embedding_identity': identity,
                                                     'chunk_count': len(chunks), 'corpus_digest': corpus['content_digest']})

        class Index:
            index = 'test-index'
            dimension = 3

            async def candidates(self, scope, query, vector, *, limit):
                if mode == "search_unavailable":
                    raise httpx.ConnectError("fixture service unreachable")
                return [{**c, 'content': '伪造索引内容'} for c in (unrelated, anchor)]

        called = []

        def rerank(query, documents):
            called.append(query)
            assert documents[1]['content'] == anchor['content']
            if mode == "model_unavailable":
                raise RuntimeError("fixture weights unavailable")
            if mode == "invalid_order":
                return [{"id": "unverified", "content": "重排器伪造内容"}]
            return [{**document, 'content': '重排器伪造内容'} for document in reversed(documents)]

        request = GenerationRequest(message='小满送信到灯塔')
        retrieve = partial(retrieve_source_context, database=db, index=Index(), provider=Provider(),
                           rerank=None if mode == "literal" else rerank)
        if mode == "search_unavailable":
            async def complete(compiled_request, session, request_id):
                assert compiled_request.retrieval.reason == "retrieval_unavailable"
                assert compiled_request.retrieval.documents == ()
                return "目前没找到原文可以核对。", "fixture"

            chat = CharacterChatService(db, complete, sources=retrieve)
            turn = asyncio.run(chat.generate('owner', session['id'], revision=1, request_id='offline', message=request.message))
            assert turn["reply"] == "目前没找到原文可以核对。"
            assert not turn["evidence"] and not called
            return

        result = asyncio.run(retrieve(request, session))
        assert called == ([] if mode == "literal" else [request.message])
        first = result.retrieval.documents[0]
        assert (first['source_id'], first['line_start'], first['line_end']) == ('original', 3, 5)
        assert first['content'] == '风声渐小。\n小满送信到灯塔。\n门卫让她进去。'
        assert result.retrieval.citations[0]['source_id'] == first['id']
        assert '伪造' not in result.retrieval.evidence and '不应跨段' not in result.retrieval.evidence
    finally:
        db.close_connection()
