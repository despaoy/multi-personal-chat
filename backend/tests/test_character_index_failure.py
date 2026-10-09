"""Character index failure and query scope are distinct runtime outcomes."""
import json
from unittest.mock import Mock

import numpy as np
import pytest
from test_multiscale_rag_runtime import FakeQueryEmbeddingProvider

from knowledge.multiscale_rag import runtime as module
from knowledge.retrieval_core.documents import KnowledgeIndexDocument

QUERY = '月社妃是谁？'


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.delenv('CHARACTER_RAG_INDEX_ROOT', raising=False)
    monkeypatch.delenv('MULTISCALE_RAG_INDEX_ROOT', raising=False)
    monkeypatch.setenv('CHARACTER_RAG_RERANKER_ENABLED', 'false')
    monkeypatch.setattr(module, 'LocalMeanPoolingEmbeddingProvider', lambda **kw: FakeQueryEmbeddingProvider())
    for directory, kind in [('card_index', 'fact'), ('scene_story_index', 'story'), ('evidence_index', 'evidence')]:
        root = tmp_path / directory
        root.mkdir()
        doc = KnowledgeIndexDocument(directory, 'tsukiyashiro_kisaki', kind, '月社妃',
                                     '月社妃在图书室阅读。', '月社妃在图书室阅读。', '月社妃在图书室阅读。')
        (root / 'documents.jsonl').write_text(json.dumps(doc.to_dict()), encoding='utf-8')
        (root / 'manifest.json').write_text('{}', encoding='utf-8')
        np.save(root / 'vectors.npy', np.eye(1, 384, dtype=np.float32))
    return module.MultiScaleRagRuntime(index_root=tmp_path)


@pytest.mark.parametrize('entry', ['is_available', 'stats', 'retrieve'])
@pytest.mark.parametrize('damage', ['missing', 'json', 'dimensions'])
def test_broken_index_raises_instead_of_unavailable_or_no_match(prepared, entry, damage):
    root = prepared.index_root / 'card_index'
    if damage == 'missing':
        (root / 'manifest.json').unlink()
        error = FileNotFoundError
    elif damage == 'json':
        (root / 'documents.jsonl').write_text('{broken', encoding='utf-8')
        error = json.JSONDecodeError
    else:
        np.save(root / 'vectors.npy', np.zeros((1, 2), dtype=np.float32))
        error = ValueError
    with pytest.raises(error):
        prepared.retrieve_with_citations(QUERY) if entry == 'retrieve' else getattr(prepared, entry)()
    assert not prepared.is_warm() and prepared._service is None and prepared._stats == {}


@pytest.mark.parametrize('query,kwargs', [('', {}), ('今天天气怎么样', {}), (QUERY, {'filters': {'knowledge_base_id': 7}}), (QUERY, {'domain_id': 'unrelated-domain'})])
def test_out_of_scope_queries_do_not_load_character_index(prepared, monkeypatch, query, kwargs):
    loader = Mock(side_effect=AssertionError('unrelated index must not load'))
    monkeypatch.setattr(module, '_load_bundle', loader)
    assert prepared.retrieve_with_citations(query, **kwargs) is None
    loader.assert_not_called()


def test_repaired_index_loads_on_next_request_and_returns_bundle(prepared):
    manifest = prepared.index_root / 'card_index' / 'manifest.json'
    original = manifest.read_bytes()
    manifest.unlink()
    with pytest.raises(FileNotFoundError):
        prepared.retrieve_with_citations(QUERY)
    manifest.write_bytes(original)
    result = prepared.retrieve_with_citations(QUERY, domain_id=prepared.config.domain_id)
    assert result is not None and result['context_trust'] == 'untrusted_retrieved_evidence'
    assert prepared.is_available() and prepared.stats()['documents'] == 3
    assert prepared.is_warm()


def test_background_failure_is_sanitized_but_request_still_raises(prepared, monkeypatch, caplog):
    error = OSError('private-source-content')
    monkeypatch.setattr(module, '_load_bundle', Mock(side_effect=error))
    calls = []

    class InlineThread:
        def __init__(self, *, target, **kwargs):
            self.target = target
        def start(self):
            calls.append(1)
            self.target()

    monkeypatch.setattr(module.threading, 'Thread', InlineThread)
    prepared.warmup_async()
    prepared.warmup_async()
    assert calls == [1] and not prepared.is_warm()
    assert 'OSError' in caplog.text and 'private-source-content' not in caplog.text
    with pytest.raises(OSError) as caught:
        prepared.retrieve_with_citations(QUERY)
    assert caught.value is error
