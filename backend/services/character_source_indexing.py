"""Publish an immutable source edition; never acknowledge partial index writes."""

import asyncio
import os

import httpx

from knowledge.character_search import CharacterSearchScope
from repositories.character_sources import CharacterSourceRepository
from services.character_chat_vectors import embedding_identity


async def publish_source_index(owner, corpus_id, *, database, index, provider, expected_attempt=None):
    repo = CharacterSourceRepository(database)
    corpus = await asyncio.to_thread(repo.get, owner, corpus_id)
    attempt = corpus['index_attempt'] if expected_attempt is None else expected_attempt
    if corpus['index_attempt'] != attempt or corpus['index_status'] == 'failed':
        raise ValueError('Source index attempt is no longer active')
    chunks = await asyncio.to_thread(repo.chunks, owner, corpus_id)
    identity = await asyncio.to_thread(embedding_identity, provider)
    if provider.dimension != index.dimension:
        raise ValueError('Search index embedding dimension mismatch')
    output = await asyncio.to_thread(provider.embed_texts, [chunk['content'] for chunk in chunks])
    vectors = [row.tolist() if hasattr(row, 'tolist') else list(row) for row in output]
    scope = CharacterSearchScope(owner, corpus['character_id'], corpus['character_revision'], corpus_id, identity)
    # A revocation during encoding must not start another index publication.
    await asyncio.to_thread(repo.get, owner, corpus_id)
    count = await index.write_chunks(scope, chunks, vectors)
    receipt = {'index_name': index.index, 'embedding_identity': identity, 'corpus_digest': corpus['content_digest'], 'chunk_count': count}
    await asyncio.to_thread(repo.mark_indexed, owner, corpus_id, receipt, expected_attempt=attempt)
    return receipt


async def publish_configured_source_index(owner, corpus_id, *, database, expected_attempt):
    from knowledge.character_search import CharacterSearchIndex
    from knowledge.retrieval_core.embedding import get_default_embedding_provider

    url = os.getenv('CHARACTER_SEARCH_URL', '').strip()
    if not url:
        raise ValueError('CHARACTER_SEARCH_URL is required for source indexing')
    provider = get_default_embedding_provider()
    username, password = os.getenv('CHARACTER_SEARCH_USERNAME'), os.getenv('CHARACTER_SEARCH_PASSWORD')
    async with httpx.AsyncClient(base_url=url, auth=(username, password or '') if username else None,
                                 timeout=60, trust_env=False, follow_redirects=False) as client:
        index = CharacterSearchIndex(client, os.getenv('CHARACTER_SEARCH_INDEX', 'character-sources-v1'), provider.dimension)
        await index.ensure_ready()
        await publish_source_index(owner, corpus_id, database=database, index=index, provider=provider,
                                   expected_attempt=expected_attempt)
