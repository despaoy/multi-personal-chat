import json

import httpx
import pytest

from character.library import CharacterSettings
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from knowledge.character_search import CharacterSearchIndex
from repositories.character_library import CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository
from services.character_source_indexing import publish_source_index


class Provider:
    model_id = 'test'
    model_fingerprint = 'weights-v1'
    dimension = 3

    def embed_texts(self, texts):
        return [[1, 0, 0] for _ in texts]


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['partial', 'revoked', 'superseded'])
async def test_only_complete_live_publication_is_marked_ready(tmp_path, failure):
    db = SQLiteDB(tmp_path / 'index-publication.db')
    repo = CharacterSourceRepository(db)
    char = repo.characters.create('owner', CharacterSettings(name='旅人', persona='谨慎', greeting='你好'))
    corpus = repo.publish('owner', char['id'], 1, [{'id': 'one', 'title': '第一章', 'content': '第一行'},
                                               {'id': 'two', 'title': '第二章', 'content': '第二行'}])
    calls = []
    fail = True

    def respond(request):
        nonlocal fail
        assert request.url.params['refresh'] == 'wait_for'
        assert request.headers['content-type'] == 'application/x-ndjson'
        assert request.content.endswith(b'\n')
        lines = [json.loads(line) for line in request.content.splitlines()]
        keys = [line['index']['_id'] for line in lines[::2]]
        calls.append(keys)
        if failure == 'revoked' and fail:
            repo.revoke('owner', corpus['id'])
        if failure == 'superseded' and fail:
            repo.set_index_job_status('owner', corpus['id'], 1, 'failed')
            repo.retry_index_job('owner', corpus['id'], 1)
        items = [{'index': {'_id': key, '_index': 'character-sources',
                            'status': 500 if failure == 'partial' and fail and i == 1 else 201}}
                 for i, key in enumerate(keys)]
        return httpx.Response(200, json={'errors': False, 'items': items})

    try:
        async with httpx.AsyncClient(base_url='http://localhost:9200', transport=httpx.MockTransport(respond)) as client:
            index = CharacterSearchIndex(client, 'character-sources', 3)
            error = {'partial': RuntimeError, 'revoked': CharacterNotFoundError, 'superseded': RevisionConflictError}[failure]
            with pytest.raises(error):
                await publish_source_index('owner', corpus['id'], database=db, index=index, provider=Provider())
            rows = db.execute_sql('SELECT index_receipt_json FROM character_source_corpora WHERE id=:id', {'id': corpus['id']})
            assert rows[0]['index_receipt_json'] is None
            if failure == 'superseded':
                current = repo.get('owner', corpus['id'])
                assert current['index_status'] == 'pending' and current['index_attempt'] == 2
            if failure == 'partial':
                fail = False
                receipt = await publish_source_index('owner', corpus['id'], database=db, index=index, provider=Provider())
                assert calls[0] == calls[1] and len(set(calls[1])) == 2
                assert receipt['chunk_count'] == 2
                repo.require_indexed('owner', corpus['id'], index.index, receipt['embedding_identity'])
                with pytest.raises(ValueError, match='索引'):
                    repo.require_indexed('owner', corpus['id'], index.index, 'different-model')
    finally:
        db.close_connection()
