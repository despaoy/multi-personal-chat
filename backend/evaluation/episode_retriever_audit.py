"""Offline lexical/embedding ranking diagnosis, not current-fact inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from evaluation.episodic_recall_audit import _terms


def cosine_scores(query, vectors):
    q, matrix = np.asarray(query, dtype=float), np.asarray(vectors, dtype=float)
    if q.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != len(q):
        raise ValueError('Embedding shape mismatch')
    if not np.isfinite(q).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite embedding')
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    if np.any(denominator == 0):
        raise ValueError('Zero embedding')
    return (matrix @ q / denominator).tolist()


def ranked(records, scores):
    if len(records) != len(scores) or not all(np.isfinite(scores)):
        raise ValueError('Invalid ranking scores')
    return sorted([dict(id=row['id'], score=float(score)) for row, score in zip(records, scores)],
                  key=lambda row: (-row['score'], row['id']))


def scoped_semantic_hits(source, query, provider, top_k):
    """Candidate hints only; same source/filter contract as the suffix reader."""
    from character.memory_extractor import memory_write_allowed

    if not 1 <= top_k <= 20:
        raise ValueError('Expected bounded top_k')
    scoped = [row for row in source.episodes if row.scope == source.scope]
    blocked = {row.session for row in scoped if (row.message and not memory_write_allowed(row.message))
               or (row.reply and not memory_write_allowed(row.reply))}
    rows = [row for row in scoped if row.session not in blocked and row.message.strip()]
    if not rows:
        return []
    vectors = provider.embed_texts([query, *(row.message for row in rows)])
    return ranked([{'id': row.source_id} for row in rows], cosine_scores(vectors[0], vectors[1:]))[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    from knowledge.retrieval_core.embedding import get_default_embedding_provider

    provider = get_default_embedding_provider()
    started = time.monotonic()
    records = suite['records']
    vectors = provider.embed_texts([row['message'] for row in records])
    encode_seconds = time.monotonic() - started
    for query in suite['queries']:
        started = time.monotonic()
        scores = cosine_scores(provider.embed_query(query['text']), vectors)
        semantic_seconds = time.monotonic() - started
        qterms = _terms(query['text'])
        rankings = dict(bigram=ranked(records, [len(qterms & _terms(row['message'])) for row in records]),
                        unigram=ranked(records, [len(set(query['text']) & set(row['message'])) for row in records]),
                        semantic=ranked(records, scores))
        result = dict(query=query, rankings=rankings, query_seconds=semantic_seconds,
                      interpretation='ranking_only_not_current_answer', synthetic=True)
        with (args.output / 'rankings.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
    manifest = dict(model_id=provider.model_id, dimension=provider.dimension,
        model_fingerprint=provider.model_fingerprint, corpus_encode_seconds=encode_seconds,
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.suite, Path(__file__))},
        generation_calls=0, runtime_enabled=False)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
