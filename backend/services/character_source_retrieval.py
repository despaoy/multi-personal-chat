"""Join search candidates to authoritative original sources for a bound conversation."""

import asyncio
import logging
import os
from dataclasses import replace

import httpx

from db.errors import RevisionConflictError
from inference.generation_request import RetrievalResult
from infra.environment import read_bool, read_positive_float
from knowledge.character_search import CharacterSearchIndex, CharacterSearchScope
from knowledge.character_source_ranking import (
    SOURCE_CANDIDATE_LIMIT,
    expand_source_context,
    rank_source_candidates,
    rerank_source_model,
)
from knowledge.evidence_packets import document_evidence_packets
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError
from repositories.character_sources import CharacterSourceRepository, SourceIndexNotReadyError
from services.character_chat_policy import require_local_endpoint
from services.character_chat_vectors import embedding_identity

logger = logging.getLogger(__name__)


class MissingSourceSearchURLError(ValueError):
    def __init__(self):
        super().__init__("原作资料检索已启用，但未配置 CHARACTER_SEARCH_URL。请联系管理员配置检索地址后重试。")


def _unavailable(request, reason='retrieval_unavailable'):
    return replace(request, retrieval=RetrievalResult(
        status='character_abstention', reason=reason,
    ))


def _revoked_branch_edition(repo, session):
    return session.get('mode') == 'branch' and repo.has_edition(
        session['owner_user_id'], session['character_id'], session['character_revision'],
        session['source_corpus_id'], revoked=True,
    )


async def retrieve_source_context(request, session, *, database, index, provider, rerank=None, rerank_timeout=None):
    corpus_id = session.get('source_corpus_id')
    if not corpus_id:
        return request
    owner = session['owner_user_id']
    repo = CharacterSourceRepository(database)
    chats = CharacterChatRepository(database)

    def validate(*, initial=False):
        current = chats.get(owner, session['id'])
        if current['revision'] != session['revision'] or current['source_corpus_id'] != corpus_id:
            raise RevisionConflictError()
        try:
            corpus = repo.completed_edition(owner, corpus_id)
        except CharacterNotFoundError:
            if initial and _revoked_branch_edition(repo, session):
                return None
            raise
        if corpus['character_id'] != session['character_id'] or corpus['character_revision'] != session['character_revision']:
            raise ValueError('Bound source edition no longer matches the character')
        if provider.dimension != index.dimension:
            raise ValueError('Source query embedding dimension mismatch')
        identity = embedding_identity(provider)
        repo.verify_index_receipt(corpus, index.index, identity)
        return corpus, identity

    try:
        validated = await asyncio.to_thread(validate, initial=True)
        if validated is None:
            return _unavailable(request, 'source_version_revoked')
    except SourceIndexNotReadyError:
        # A newly published or copied corpus may still be indexing. Admit
        # no source text or citations, while allowing an in-character reply.
        return _unavailable(request)
    _, identity = validated
    vector = await asyncio.to_thread(provider.embed_query, request.message)
    vector = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
    scope = CharacterSearchScope(owner, session['character_id'], session['character_revision'], corpus_id, identity)
    try:
        candidates = await index.candidates(scope, request.message, vector, limit=SOURCE_CANDIDATE_LIMIT)
    except httpx.HTTPError as exc:
        logger.warning("Character source search unavailable type=%s", type(exc).__name__)
        return _unavailable(request)
    documents = await asyncio.to_thread(repo.verify_candidates, owner, session['character_id'],
                                        session['character_revision'], corpus_id, candidates)
    ranked = await rank_source_candidates(request.message, documents, rerank=rerank, timeout=rerank_timeout)
    corpus, _ = await asyncio.to_thread(validate)
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
    if session.get('mode') == 'branch' and await asyncio.to_thread(
        _revoked_branch_edition, CharacterSourceRepository(database), session,
    ):
        return _unavailable(request, 'source_version_revoked')
    from knowledge.retrieval_core.embedding import get_default_embedding_provider

    url = os.getenv('CHARACTER_SEARCH_URL', '').strip()
    if not url:
        raise MissingSourceSearchURLError()
    require_local_endpoint(session, url, service="原作资料检索")
    provider = get_default_embedding_provider()
    rerank = rerank_source_model if read_bool(os.environ, 'RERANKER_ENABLED') else None
    timeout = read_positive_float(os.environ, 'CHARACTER_SOURCE_RERANK_TIMEOUT_SECONDS', 25.0) if rerank else None
    username, password = os.getenv('CHARACTER_SEARCH_USERNAME'), os.getenv('CHARACTER_SEARCH_PASSWORD')
    async with httpx.AsyncClient(base_url=url, auth=(username, password or '') if username else None,
                                 timeout=30, trust_env=False, follow_redirects=False) as client:
        index = CharacterSearchIndex(client, os.getenv('CHARACTER_SEARCH_INDEX', 'character-sources-v1'), provider.dimension)
        return await retrieve_source_context(request, session, database=database, index=index, provider=provider,
                                             rerank=rerank, rerank_timeout=timeout)
