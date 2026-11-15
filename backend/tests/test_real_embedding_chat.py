"""Replay freshly generated real model vectors through PostgreSQL and chat prompts.

The generation completion is controlled; this does not validate cloud-model answers.
"""

import asyncio
import json
import os
from functools import partial
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.getenv('REAL_EMBEDDING_FIXTURE'), reason='fresh real embedding output required')
def test_real_embeddings_retrieve_private_memory_for_chat():
    from character.library import CharacterSettings
    from db.pg_database import PgDatabase, SyncPgAdapter
    from repositories.character_chat import CharacterChatRepository
    from repositories.character_library import CharacterNotFoundError
    from services.character_chat import CharacterChatService
    from services.character_chat_vectors import select_memories

    data = json.loads(Path(os.environ['REAL_EMBEDDING_FIXTURE']).read_text())['result']
    assert data['device'] == 'cpu' and data['dimension'] == 384

    class RecordedProvider:
        model_id = data['model_id']
        model_fingerprint = data['fingerprint']
        dimension = data['dimension']

        def embed_texts(self, texts):
            return [data['vectors'][data['memories'].index(text)] for text in texts]

        def embed_query(self, text):
            return data['query_vectors'][data['queries'].index(text)]

    db = SyncPgAdapter(PgDatabase(os.environ['DATABASE_URL']))
    try:
        chats = CharacterChatRepository(db)
        character = chats.characters.create('real-embedding-owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
        session = chats.create('real-embedding-owner', character['id'], '真实模型记忆验收')
        for content in data['memories']:
            session = chats.manage_memory(session['owner_user_id'], session['id'], session['revision'], content=content)
        retrieve = partial(select_memories, database=db, provider=RecordedProvider(), limit=1)
        for i, query in enumerate(data['queries']):
            async def complete(request, _session, _request, expected=data['memories'][i]):
                context = json.loads(request.character_context.conversation_reference_context)
                assert [m['content'] for m in context['memories']] == [expected]
                return '已读取对应记忆', 'controlled-completion'

            service = CharacterChatService(db, complete, retrieve=retrieve)
            asyncio.run(service.generate(session['owner_user_id'], session['id'], revision=session['revision'],
                                         request_id=f'real-vector-{i}', message=query))
            session = chats.get(session['owner_user_id'], session['id'])
            assert len(session['state']['memories']) == 4
        forged = dict(session, owner_user_id='other-account')
        with pytest.raises(CharacterNotFoundError):
            retrieve(forged, data['queries'][0])
        memory = session['state']['memories'][0]
        session = chats.manage_memory(session['owner_user_id'], session['id'], session['revision'],
                                      memory_id=memory['id'])
        assert all(m['id'] != memory['id'] for m in retrieve(session, data['queries'][0]))
        chats.delete_sessions(session['owner_user_id'], [{'id': session['id'], 'revision': session['revision']}])
    finally:
        db.close()
