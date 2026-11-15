import hashlib
import json

import httpx
import pytest

from knowledge.character_search import CharacterSearchIndex, CharacterSearchScope, index_definition

SCOPE = CharacterSearchScope('owner', 'character', 1, 'corpus-v1', 'embedding-v1')


def hit(identity):
    return {'_source': {**SCOPE.fields(), 'chunk_id': identity, 'source_id': 'chapter1',
                        'content_sha256': hashlib.sha256(b'body').hexdigest(), 'line_start': 1, 'line_end': 2,
                        'content': 'must not be trusted', 'title': 'untrusted title'}}


@pytest.mark.asyncio
async def test_hybrid_search_filters_both_channels_and_returns_metadata_only():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        hits = [hit('b'), hit('a')] if 'knn' in body['query'] else [hit('a'), hit('b')]
        return httpx.Response(200, json={'timed_out': False, '_shards': {'failed': 0}, 'hits': {'hits': hits}})

    async with httpx.AsyncClient(base_url='http://localhost:9200', transport=httpx.MockTransport(respond)) as client:
        results = await CharacterSearchIndex(client, 'character-sources', 3).candidates(SCOPE, '剧情', [1, 0, 0])
    assert [r['chunk_id'] for r in results] == ['a', 'b']
    assert all('content' not in r and 'title' not in r for r in results)
    expected = [{'term': {k: v}} for k, v in SCOPE.fields().items()]
    assert requests[0]['query']['bool']['filter'] == expected
    assert requests[1]['query']['knn']['embedding']['filter']['bool']['filter'] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['scope', 'duplicate', 'partial', 'timeout', 'lines', 'digest'])
async def test_bad_or_partial_search_results_fail_closed(failure):
    def respond(request):
        item = hit('a')
        if failure == 'scope':
            item['_source']['owner_id'] = 'other'
        if failure == 'lines':
            item['_source']['line_start'] = 0
        if failure == 'digest':
            item['_source']['content_sha256'] = 'invalid'
        return httpx.Response(200, json={'timed_out': failure == 'timeout', '_shards': {'failed': int(failure == 'partial')},
                                        'hits': {'hits': [item, item] if failure == 'duplicate' else [item]}})

    async with httpx.AsyncClient(base_url='http://localhost:9200', transport=httpx.MockTransport(respond)) as client:
        with pytest.raises((ValueError, RuntimeError)):
            await CharacterSearchIndex(client, 'character-sources', 3).candidates(SCOPE, '剧情', [1, 0, 0])


def test_mapping_uses_strict_scope_and_filtered_lucene_vectors():
    mapping = index_definition(384)['mappings']
    assert mapping['dynamic'] == 'strict'
    assert mapping['properties']['owner_id']['type'] == 'keyword'
    assert mapping['properties']['embedding']['method']['engine'] == 'lucene'
    with pytest.raises(ValueError):
        CharacterSearchIndex(None, '../_all', 384)
