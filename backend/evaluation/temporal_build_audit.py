"""Compare frozen and revised builders on identical source data, without vectors."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from knowledge.multiscale_rag.constants import INDEX_FORMAT_VERSION
from knowledge.multiscale_rag.index_builder import CharacterKnowledgeIndexBuilder
from knowledge.retrieval_core.registry import get_default_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--baseline-builder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    name = 'knowledge.multiscale_rag._audit_baseline'
    spec = importlib.util.spec_from_file_location(name, args.baseline_builder)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    config = get_default_registry().require('tsukiyashiro_kisaki')
    options = dict(domain_id=config.domain_id, index_version=INDEX_FORMAT_VERSION,
                   aliases=config.aliases, corpus_root=args.repo_root)
    before = module.CharacterKnowledgeIndexBuilder(**options).build(config.source_root, args.scenes)
    after = CharacterKnowledgeIndexBuilder(**options).build(config.source_root, args.scenes)
    old = {d.id: d.to_dict() for d in before.documents}
    new = {d.id: d.to_dict() for d in after.documents}
    if old.keys() != new.keys():
        raise ValueError('Unexpected document inventory change')
    changes = []
    for key, item in new.items():
        prior = old[key]
        if prior == item:
            continue
        prior_other = {k: v for k, v in prior.items() if k not in {'temporal_scope', 'metadata'}}
        new_other = {k: v for k, v in item.items() if k not in {'temporal_scope', 'metadata'}}
        old_meta = {k: v for k, v in prior['metadata'].items() if k != 'semantic_temporal_scope'}
        new_meta = {k: v for k, v in item['metadata'].items() if k != 'semantic_temporal_scope'}
        if prior_other != new_other or old_meta != new_meta:
            raise ValueError(f'Unexpected non-temporal change: {key}')
        changes.append({'id': key, 'before': prior, 'after': item})
    paths = {args.scenes.resolve(), args.baseline_builder.resolve(),
             Path(sys.modules[CharacterKnowledgeIndexBuilder.__module__].__file__).resolve()}
    paths.update(p.resolve() for p in config.source_root.rglob('*.jsonl'))
    for doc in after.documents:
        if doc.source.source_path:
            path = (args.repo_root / doc.source.source_path).resolve()
            if path.is_file():
                paths.add(path)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    summary = {'documents': len(new), 'changed': len(changes), 'counts': after.counts,
               'exact_evidence_matches': after.exact_evidence_matches,
               'embedding_text_unchanged': True, 'source_and_content_unchanged': True,
               'source_hashes': hashes, 'scope': 'document_build_only_no_index_replacement_no_model'}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'changes.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in changes), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_hashes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
