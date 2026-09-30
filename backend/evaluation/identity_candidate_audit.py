"""Read-only retrieval-stage audit; never installed in the application process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge.entity_scope import in_identity_scope
from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime


def describe(document):
    return {'id': document.id, 'type': document.document_type, 'title': document.title,
            'summary': document.summary, 'metadata': document.metadata,
            'source': document.source.to_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identity-coverage', action='store_true')
    args = parser.parse_args()
    queries = json.loads(args.queries.read_text(encoding='utf-8'))
    if not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError('Expected nonempty query strings')
    runtime = MultiScaleRagRuntime()
    if not runtime._ensure_loaded():
        raise RuntimeError('Index unavailable')
    service = runtime._service
    service.identity_coverage = args.identity_coverage
    stage = {}
    # Observe original calls without changing recall/ranking parameters or results.
    for route, retriever in service.retrievers.items():
        original = retriever.search

        def observe_search(*a, _original=original, _route=route, **kw):
            candidates = _original(*a, **kw)
            stage['route'] = sorted(_route)
            stage['recalled'] = [describe(c.document) for c in candidates]
            return candidates

        retriever.search = observe_search
    original_rank = service.reranker.rerank

    def observe_rank(analysis, candidates, **kw):
        stage['scoped_ids'] = [c.document.id for c in candidates]
        ranked = original_rank(analysis, candidates, **kw)
        stage['ranked'] = [dict(describe(c.document), score=c.rerank_score,
                                method=c.rerank_method) for c in ranked]
        return ranked

    service.reranker.rerank = observe_rank
    with args.output.open('x', encoding='utf-8') as stream:
        for query in queries:
            stage.clear()
            result = service.retrieve(query, top_k=3)
            subject = result['identity_scope']['subject']
            inventory = [describe(d) for d in service.by_id.values()
                         if subject and d.document_type == 'relation' and in_identity_scope(d, subject)]
            stream.write(json.dumps({'query': query, **stage, 'result': result,
                                     'identity_relation_inventory': inventory}, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({'query': query, 'recalled': len(stage.get('recalled', [])),
                              'scoped': len(stage.get('scoped_ids', [])),
                              'relation_inventory': len(inventory)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
