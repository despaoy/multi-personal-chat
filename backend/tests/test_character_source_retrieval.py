import asyncio
import re
from functools import partial

import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from inference.generation_request import generate_character_response
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository
from services.character_chat import CharacterChatService
from services.character_chat_evidence import CharacterChatCompletion, evidence_snapshot
from services.character_chat_vectors import embedding_identity
from services.character_source_retrieval import retrieve_source_context


class Provider:
    model_id = 'test'
    model_fingerprint = 'test-weights'
    dimension = 3

    def embed_query(self, message):
        return [1, 0, 0]


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

            async def candidates(self, scope, query, vector):
                assert scope.owner_id == 'owner' and scope.corpus_revision == corpus['id']
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
