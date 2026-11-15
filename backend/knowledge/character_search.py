"""OpenSearch candidate retrieval; candidates are never authoritative source text.

Lucene filtered k-NN: https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/
"""

import asyncio
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CharacterSearchScope:
    owner_id: str
    character_id: str
    character_revision: int
    corpus_revision: str
    embedding_identity: str

    def fields(self):
        values = asdict(self)
        if type(self.character_revision) is not int or self.character_revision < 1:
            raise ValueError('Invalid character revision')
        if any(not isinstance(v, str) or not v.strip() for k, v in values.items() if k != 'character_revision'):
            raise ValueError('Search scope must be complete')
        return values


def index_definition(dimension):
    if type(dimension) is not int or not 1 <= dimension <= 16000:
        raise ValueError('Invalid embedding dimension')
    fields = {name: {'type': 'keyword'} for name in (
        'owner_id', 'character_id', 'corpus_revision', 'embedding_identity', 'chunk_id', 'source_id', 'content_sha256',
    )}
    fields.update(character_revision={'type': 'integer'}, title={'type': 'text'}, content={'type': 'text', 'analyzer': 'cjk'},
                  line_start={'type': 'integer'}, line_end={'type': 'integer'}, embedding={
                      'type': 'knn_vector', 'dimension': dimension,
                      'method': {'name': 'hnsw', 'engine': 'lucene', 'space_type': 'cosinesimil'},
                  })
    return {'settings': {'index': {'knn': True, 'number_of_shards': 1, 'number_of_replicas': 0}},
            'mappings': {'dynamic': 'strict', 'properties': fields}}


class CharacterSearchIndex:
    """Caller owns authenticated HTTP client and authoritative document verification."""

    def __init__(self, client, index, dimension):
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,100}', index):
            raise ValueError('Invalid dedicated search index name')
        self.client, self.index, self.dimension = client, index, dimension
        self.mapping = index_definition(dimension)

    async def create(self):
        response = await self.client.put('/' + self.index, json=self.mapping)
        response.raise_for_status()
        if response.json().get('acknowledged') is not True:
            raise RuntimeError('Search index creation was not acknowledged')

    async def ensure_ready(self):
        """Create a dedicated index once; refuse incompatible existing mappings."""
        response = await self.client.get('/' + self.index + '/_mapping')
        if response.status_code == 404:
            created = await self.client.put('/' + self.index, json=self.mapping)
            if created.status_code == 400 and created.json().get('error', {}).get('type') == 'resource_already_exists_exception':
                pass  # Another worker created it concurrently; validate its mapping below.
            else:
                created.raise_for_status()
                if created.json().get('acknowledged') is not True:
                    raise RuntimeError('Search index creation was not acknowledged')
            response = await self.client.get('/' + self.index + '/_mapping')
        response.raise_for_status()
        mappings = response.json().get(self.index, {}).get('mappings', {})
        method = mappings.get('properties', {}).get('embedding', {}).get('method', {})
        if method.get('parameters') == {}:
            method.pop('parameters')  # OpenSearch serializes the default empty HNSW options.
        if mappings != self.mapping['mappings']:
            raise ValueError('Existing source index mapping does not match configured embeddings')

    async def write_chunks(self, scope, chunks, vectors, *, batch_size=50):
        """Idempotent bulk writes; every item must be acknowledged before success."""
        fields = scope.fields()
        if type(batch_size) is not int or not 1 <= batch_size <= 100 or not chunks or len(chunks) != len(vectors):
            raise ValueError('Invalid source index batch')
        documents, identities = [], set()
        for chunk, vector in zip(chunks, vectors, strict=True):
            if any(chunk.get(key) != value for key, value in fields.items() if key != 'embedding_identity'):
                raise ValueError('Source chunk scope mismatch')
            identity = chunk.get('chunk_id')
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError('Duplicate or missing source chunk identity')
            identities.add(identity)
            if not isinstance(chunk.get('content'), str) or hashlib.sha256(chunk['content'].encode()).hexdigest() != chunk.get('content_sha256'):
                raise ValueError('Source chunk content mismatch')
            if len(vector) != self.dimension or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector) or not any(vector):
                raise ValueError('Invalid source embedding')
            doc = {key: chunk[key] for key in ('chunk_id', 'source_id', 'title', 'content', 'content_sha256', 'line_start', 'line_end')}
            doc.update(fields, embedding=vector)
            key = hashlib.sha256(json.dumps([fields, identity], sort_keys=True).encode()).hexdigest()
            documents.append((key, doc))
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            lines = []
            for key, document in batch:
                lines.extend([json.dumps({'index': {'_index': self.index, '_id': key}}), json.dumps(document, ensure_ascii=False, allow_nan=False)])
            response = await self.client.post('/_bulk', params={'refresh': 'wait_for'},
                                              content=('\n'.join(lines) + '\n').encode('utf-8'),
                                              headers={'Content-Type': 'application/x-ndjson'})
            response.raise_for_status()
            body = response.json()
            items = body.get('items', [])
            if body.get('errors') is not False or len(items) != len(batch):
                raise RuntimeError('Source indexing was incomplete')
            for (key, _), item in zip(batch, items, strict=True):
                result = item.get('index', {})
                if result.get('_id') != key or result.get('_index') != self.index or result.get('status') not in {200, 201} or result.get('error'):
                    raise RuntimeError('Source indexing item was not acknowledged')
        return len(documents)

    async def candidates(self, scope, query, vector, *, limit=8, candidate_count=40):
        fields = scope.fields()
        if not isinstance(query, str) or not query.strip() or len(query) > 8000:
            raise ValueError('Invalid source query')
        if type(limit) is not int or type(candidate_count) is not int or not 1 <= limit <= candidate_count <= 100:
            raise ValueError('Invalid source candidate budget')
        if len(vector) != self.dimension or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
            raise ValueError('Invalid source query vector')
        if not any(x != 0 for x in vector):
            raise ValueError('Zero source query vector')
        filters = [{'term': {key: value}} for key, value in fields.items()]
        queries = [
            {'bool': {'filter': filters, 'must': [{'multi_match': {'query': query, 'fields': ['title^2', 'content']}}]}},
            {'knn': {'embedding': {'vector': vector, 'k': candidate_count, 'filter': {'bool': {'filter': filters}}}}},
        ]

        async def search(body):
            response = await self.client.post('/' + self.index + '/_search',
                                              json={'size': candidate_count, '_source': {'excludes': ['embedding', 'content']}, 'query': body})
            response.raise_for_status()
            data = response.json()
            if data.get('timed_out') is not False or data.get('_shards', {}).get('failed') != 0:
                raise RuntimeError('Source search was incomplete')
            return data['hits']['hits']

        ranked = await asyncio.gather(*(search(body) for body in queries))
        records, scores = {}, {}
        for hits in ranked:
            seen = set()
            for position, hit in enumerate(hits, 1):
                source = hit['_source']
                if any(source.get(key) != value for key, value in fields.items()):
                    raise ValueError('Search candidate escaped its authorized scope')
                identity = source.get('chunk_id')
                if not isinstance(identity, str) or not identity or identity in seen:
                    raise ValueError('Invalid or duplicate source chunk identity')
                seen.add(identity)
                if not isinstance(source.get('source_id'), str) or not source['source_id']:
                    raise ValueError('Source candidate has no original identity')
                if not re.fullmatch(r'[0-9a-f]{64}', source.get('content_sha256', '')):
                    raise ValueError('Source candidate has no content fingerprint')
                start, end = source.get('line_start'), source.get('line_end')
                if type(start) is not int or type(end) is not int or not 1 <= start <= end:
                    raise ValueError('Invalid source line range')
                # Only these metadata fields may cross the search boundary. Reload text from primary storage.
                record = {key: source[key] for key in (*fields, 'chunk_id', 'source_id', 'content_sha256', 'line_start', 'line_end')}
                if identity in records and records[identity] != record:
                    raise ValueError('Source candidate changed between retrieval channels')
                records[identity] = record
                scores[identity] = scores.get(identity, 0) + 1 / (60 + position)
        return [dict(records[key], fusion_score=scores[key]) for key in sorted(records, key=lambda key: (-scores[key], key))[:limit]]
