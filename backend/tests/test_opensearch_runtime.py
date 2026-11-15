"""Opt-in real OpenSearch check; creates and deletes only its own unique index."""

import os
import uuid

import httpx
import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from knowledge.character_search import CharacterSearchIndex, CharacterSearchScope
from repositories.character_library import CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository
from services.character_source_indexing import publish_source_index


@pytest.mark.skipif(not os.getenv('TEST_OPENSEARCH_URL'), reason='isolated real OpenSearch required')
@pytest.mark.asyncio
async def test_real_opensearch_publication_hybrid_search_and_revocation(tmp_path):
    db = SQLiteDB(tmp_path / 'real-search.db')
    repo = CharacterSourceRepository(db)
    character = repo.characters.create('search-owner', CharacterSettings(name='小满', persona='灯塔守望者', greeting='你好'))
    corpus = repo.publish('search-owner', character['id'], 1, [
        {'id': 'chapter1', 'title': '灯塔之夜', 'content': '小满把信送到了灯塔。'},
        {'id': 'chapter2', 'title': '花园清晨', 'content': '阿青在花园修剪玫瑰。'},
    ])

    class Provider:
        model_id = 'runtime-test'
        model_fingerprint = 'controlled-vectors'
        dimension = 3

        def embed_texts(self, texts):
            return [[1, 0, 0] if '灯塔' in text else [0, 1, 0] for text in texts]

    name = 'lhm-validation-' + uuid.uuid4().hex
    try:
        async with httpx.AsyncClient(base_url=os.environ['TEST_OPENSEARCH_URL'], timeout=60,
                                     trust_env=False, follow_redirects=False) as client:
            index = CharacterSearchIndex(client, name, 3)
            try:
                await index.ensure_ready()
                await index.ensure_ready()
                receipt = await publish_source_index('search-owner', corpus['id'], database=db, index=index, provider=Provider())
                assert receipt['chunk_count'] == 2
                repo.require_indexed('search-owner', corpus['id'], name, receipt['embedding_identity'])
                scope = CharacterSearchScope('search-owner', character['id'], 1, corpus['id'], receipt['embedding_identity'])
                candidates = await index.candidates(scope, '谁把信送到了灯塔？', [1, 0, 0], limit=1)
                assert len(candidates) == 1
                source = repo.verify_candidate('search-owner', character['id'], 1, corpus['id'], candidates[0])
                assert source['title'] == '灯塔之夜' and source['content'] == '小满把信送到了灯塔。'
                other = CharacterSearchScope('other-account', character['id'], 1, corpus['id'], receipt['embedding_identity'])
                assert await index.candidates(other, '灯塔', [1, 0, 0]) == []
                await publish_source_index('search-owner', corpus['id'], database=db, index=index, provider=Provider())
                response = await client.get('/' + name + '/_count')
                response.raise_for_status()
                assert response.json()['count'] == 2
                repo.revoke('search-owner', corpus['id'])
                with pytest.raises(CharacterNotFoundError):
                    repo.verify_candidate('search-owner', character['id'], 1, corpus['id'], candidates[0])
            finally:
                response = await client.delete('/' + name)
                if response.status_code != 404:
                    response.raise_for_status()
    finally:
        db.close_connection()
