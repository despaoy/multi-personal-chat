"""Real vector replay + real search/database + production chat/citation pipeline."""

import json
import os
import re
import uuid
from functools import partial
from pathlib import Path

import httpx
import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from inference.generation_request import generate_character_response
from knowledge.character_search import CharacterSearchIndex
from repositories.character_chat import CharacterChatRepository
from repositories.character_sources import CharacterSourceRepository
from services.character_chat import CharacterChatService
from services.character_chat_evidence import CharacterChatCompletion, evidence_snapshot
from services.character_source_indexing import publish_source_index
from services.character_source_retrieval import retrieve_source_context


@pytest.mark.skipif(not (os.getenv('TEST_OPENSEARCH_URL') and os.getenv('REAL_EMBEDDING_FIXTURE')), reason='real search and fresh embeddings required')
@pytest.mark.asyncio
async def test_real_sources_reach_citable_chat_and_survive_branch(tmp_path):
    data = json.loads(Path(os.environ['REAL_EMBEDDING_FIXTURE']).read_text())['result']

    class Provider:
        model_id = data['model_id']
        model_fingerprint = data['fingerprint']
        dimension = data['dimension']

        def embed_texts(self, texts):
            return [data['vectors'][data['memories'].index(text)] for text in texts]

        def embed_query(self, message):
            return data['query_vectors'][data['queries'].index(message)]

    if os.getenv('RUN_CHARACTER_PG_INTEGRATION') == '1':
        from db.pg_database import PgDatabase, SyncPgAdapter

        db = SyncPgAdapter(PgDatabase(os.environ['DATABASE_URL']))
    else:
        db = SQLiteDB(tmp_path / 'real-source-stack.db')
    created = False
    name = 'lhm-stack-' + uuid.uuid4().hex
    try:
        sources, chats = CharacterSourceRepository(db), CharacterChatRepository(db)
        char = sources.characters.create('stack-owner', CharacterSettings(name='资料助手', persona='依据资料回应', greeting='你好'))
        corpus = sources.publish('stack-owner', char['id'], 1, [
            {'id': f'doc-{i}', 'title': f'验收资料{i + 1}', 'content': content} for i, content in enumerate(data['memories'])
        ])
        session = chats.create('stack-owner', char['id'], '真实资料链路')
        async with httpx.AsyncClient(base_url=os.environ['TEST_OPENSEARCH_URL'], timeout=60,
                                     trust_env=False, follow_redirects=False) as client:
            index = CharacterSearchIndex(client, name, Provider.dimension)
            try:
                await index.create()
                created = True
                await publish_source_index('stack-owner', corpus['id'], database=db, index=index, provider=Provider())
                for i, message in enumerate(data['queries']):
                    async def complete(request, _session, _request_id, expected=data['memories'][i]):
                        assert request.retrieval.documents[0]['content'] == expected

                        async def generate(**kwargs):
                            prompt = str(kwargs)
                            assert expected in prompt
                            namespace = re.search(r'\[\[cite:([0-9a-f]{12}):S1\]\]', prompt).group(1)
                            return expected + f'[[cite:{namespace}:S1]]'

                        result = await generate_character_response(request, generate)
                        return CharacterChatCompletion(result.reply, 'controlled-answer', evidence_snapshot(result))

                    service = CharacterChatService(db, complete, sources=partial(retrieve_source_context, database=db, index=index, provider=Provider()))
                    turn = await service.generate('stack-owner', session['id'], revision=session['revision'],
                                                  request_id=f'question-{i}', message=message)
                    assert [source['excerpt'] for source in turn['evidence']['sources']] == [data['memories'][i]]
                    session = chats.get('stack-owner', session['id'])
                saved = chats.turns('stack-owner', session['id'])
                assert len(saved) == 4
                branch = chats.create('stack-owner', char['id'], '来源快照分支', source_session_id=session['id'],
                                      source_revision=session['revision'], source_turn_id=saved[1]['id'], hypothesis='假设另一个结局')
                assert branch['source_corpus_id'] == corpus['id']
                assert branch['baseline']['history'][1]['evidence'] == saved[1]['evidence']
            finally:
                if created:
                    response = await client.delete('/' + name)
                    response.raise_for_status()
    finally:
        if isinstance(db, SQLiteDB):
            db.close_connection()
        else:
            db.close()
