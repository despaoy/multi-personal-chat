"""Join search candidates to authoritative original sources for a bound conversation."""

import asyncio
import os
from dataclasses import replace

import httpx

from db.errors import RevisionConflictError
from inference.generation_request import RetrievalResult
from infra.environment import read_bool
from knowledge.character_search import CharacterSearchIndex, CharacterSearchScope
from knowledge.character_source_ranking import SOURCE_CANDIDATE_LIMIT, expand_source_context, rank_verified_sources
from knowledge.evidence_packets import document_evidence_packets
from repositories.character_chat import CharacterChatRepository
from repositories.character_sources import CharacterSourceRepository, SourceIndexNotReadyError
from services.character_chat_policy import require_local_endpoint
from services.character_chat_vectors import embedding_identity


async def retrieve_source_context(request, session, *, database, index, provider, reranker=None):
    corpus_id = session.get('source_corpus_id')
    if not corpus_id:
        return request
    owner = session['owner_user_id']
    repo = CharacterSourceRepository(database)
    chats = CharacterChatRepository(database)
    identity = await asyncio.to_thread(embedding_identity, provider)
    if provider.dimension != index.dimension:
        raise ValueError('Source query embedding dimension mismatch')

    def validate():
        current = chats.get(owner, session['id'])
        if current['revision'] != session['revision'] or current['source_corpus_id'] != corpus_id:
            raise RevisionConflictError()
        corpus = repo.require_indexed(owner, corpus_id, index.index, identity)
        if corpus['character_id'] != session['character_id'] or corpus['character_revision'] != session['character_revision']:
            raise ValueError('Bound source edition no longer matches the character')
        return corpus

    try:
        await asyncio.to_thread(validate)
    except SourceIndexNotReadyError:
        # A newly published or copied corpus may still be indexing. Admit
        # no source text or citations, while allowing an in-character reply.
        return replace(request, retrieval=RetrievalResult(
            status='character_abstention', reason='retrieval_unavailable',
        ))
    vector = await asyncio.to_thread(provider.embed_query, request.message)
    vector = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
    scope = CharacterSearchScope(owner, session['character_id'], session['character_revision'], corpus_id, identity)
    candidates = await index.candidates(scope, request.message, vector, limit=SOURCE_CANDIDATE_LIMIT)
    documents = await asyncio.to_thread(repo.verify_candidates, owner, session['character_id'],
                                        session['character_revision'], corpus_id, candidates)
    ranked = await asyncio.to_thread(rank_verified_sources, request.message, documents, reranker=reranker)
    corpus = await asyncio.to_thread(validate)
    documents = await asyncio.to_thread(expand_source_context, ranked, corpus)
    packets = document_evidence_packets(documents)
    retrieval = RetrievalResult(status='ok' if documents else 'character_abstention',
                                evidence='\n\n'.join(p['text'] for p in packets), evidence_packets=packets,
                                documents=tuple(documents), citations=tuple(
                                    {'source_id': d['id'], 'source_title': d['title'], 'line_start': d['line_start'], 'line_end': d['line_end']}
                                    for d in documents), reason='' if documents else 'no_matching_original_sources')

    async def revalidate(current_request):
        await asyncio.to_thread(validate)
        return current_request

    return replace(request, retrieval=retrieval, public_context_revalidator=revalidate)


async def retrieve_configured_sources(request, session, *, database):
    if not session.get('source_corpus_id'):
        return request
    from knowledge.retrieval_core.embedding import get_default_embedding_provider

    url = os.getenv('CHARACTER_SEARCH_URL', '').strip()
    if not url:
        raise ValueError('CHARACTER_SEARCH_URL is required for original-source retrieval')
    require_local_endpoint(session, url)
    provider = get_default_embedding_provider()
    reranker = None
    if read_bool(os.environ, 'RERANKER_ENABLED'):
        from knowledge.reranker import get_reranker

        reranker = get_reranker()
    username, password = os.getenv('CHARACTER_SEARCH_USERNAME'), os.getenv('CHARACTER_SEARCH_PASSWORD')
    async with httpx.AsyncClient(base_url=url, auth=(username, password or '') if username else None,
                                 timeout=30, trust_env=False, follow_redirects=False) as client:
        index = CharacterSearchIndex(client, os.getenv('CHARACTER_SEARCH_INDEX', 'character-sources-v1'), provider.dimension)
        return await retrieve_source_context(request, session, database=database, index=index, provider=provider,
                                             reranker=reranker)
