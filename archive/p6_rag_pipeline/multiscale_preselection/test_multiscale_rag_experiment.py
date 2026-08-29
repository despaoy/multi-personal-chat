"""Tests for the opt-in multi-scale RAG experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from knowledge.multiscale_rag import (
    ABCase,
    MultiScaleRagService,
    OriginalTextExtractor,
    ReprocessedMultiScaleBuilder,
    ReprocessedMultiScaleService,
    RoutedMultiScaleService,
    build_multiscale_index,
    evaluate_ab,
)
from knowledge.multiscale_rag.vector_runtime import attach_vectors
from knowledge.multiscale_rag.service_v3 import CARD_TYPES, analyze_explicit_domain, choose_card_types
from knowledge.rag_pipeline.index import DomainIndex
from knowledge.rag_pipeline.query import QueryAnalyzer
from knowledge.rag_pipeline.loaders import AliasEntityNormalizer, ApprovedCardsLoader
from knowledge.rag_pipeline.registry import KnowledgeDomainConfig


class FakeEmbeddingProvider:
    dimension = 8
    model_id = "multiscale-test"
    model_fingerprint = "multiscale-test-fp"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        raw = np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[:8], dtype=np.uint8).astype(np.float32)
        return raw / np.linalg.norm(raw)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)


@pytest.fixture(autouse=True)
def fake_faiss(monkeypatch):
    """Keep the experiment tests runnable in the lightweight CI image."""

    class IndexFlatIP:
        def __init__(self, dimension: int):
            self.d = dimension
            self.matrix = np.zeros((0, dimension), dtype=np.float32)

        @property
        def ntotal(self) -> int:
            return int(self.matrix.shape[0])

        def add(self, matrix: np.ndarray) -> None:
            self.matrix = np.asarray(matrix, dtype=np.float32)

        def search(self, query: np.ndarray, top_k: int):
            scores = np.asarray(query, dtype=np.float32) @ self.matrix.T
            rows = np.argsort(-scores, axis=1)[:, :top_k]
            return np.take_along_axis(scores, rows, axis=1), rows

    def write_index(_: IndexFlatIP, path: str) -> None:
        Path(path).write_bytes(b"fake-faiss-index")

    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(IndexFlatIP=IndexFlatIP, write_index=write_index))


@pytest.fixture()
def experiment_inputs(tmp_path: Path):
    corpus_root = tmp_path / "corpus"
    source_file = corpus_root / "gametext" / "test" / "story.txt"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "开场。\n"
        "甲和乙在教室见面。\n"
        "[甲] 「乙是我的旧时好友。」\n"
        "[乙] 「我们认识很多年了。」\n"
        "两人继续交谈。\n"
        "场景结束。\n",
        encoding="utf-8",
    )

    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    relation = {
        "id": "rel_test_friend",
        "document_type": "relation",
        "title": "甲与乙的关系",
        "subject": "甲",
        "relation": "朋友",
        "target": "乙",
        "summary": "甲与乙是旧时好友",
        "evidence_text": "[甲] 「乙是我的旧时好友。」\n[乙] 「我们认识很多年了。」",
        "story": {
            "volume_number": 1,
            "story_unit_id": "vol01_test",
            "story_title": "测试故事",
            "viewpoint": "甲第一人称",
            "content_scope": "main_story",
            "temporal_scope": "current",
        },
        "source": {
            "source_path": "gametext/test/story.txt",
            "line_start": 1,
            "line_end": 6,
        },
        "reality_status": "objective",
        "review_status": "approved",
    }
    for name, records in (
        ("facts_approved.jsonl", []),
        ("relations_approved.jsonl", [relation]),
        ("events_approved.jsonl", []),
    ):
        (approved_root / name).write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    scene_path = tmp_path / "enriched_scenes.jsonl"
    scene = {
        "id": "scene_test_001",
        "document_type": "scene",
        "title": "甲与乙在教室交谈",
        "text": source_file.read_text(encoding="utf-8").rstrip("\n"),
        "story": relation["story"],
        "source": relation["source"],
        "speakers": ["甲", "乙"],
        "mentioned_characters": ["甲", "乙"],
        "present_characters": ["甲", "乙"],
        "reality_status": "objective",
        "review_status": "approved",
    }
    scene_path.write_text(json.dumps(scene, ensure_ascii=False) + "\n", encoding="utf-8")

    aliases = {"甲": "甲", "乙": "乙"}
    base_config = KnowledgeDomainConfig(
        domain_id="baseline_test",
        source_root=approved_root,
        loader=ApprovedCardsLoader(
            domain_id="baseline_test",
            index_version="v1",
            entity_normalizer=AliasEntityNormalizer(aliases),
        ),
        aliases=aliases,
        story_titles=["测试故事"],
        index_root=tmp_path / "baseline_index",
    )
    return corpus_root, approved_root, scene_path, base_config


def test_builds_separate_parent_child_hierarchy_and_tightens_evidence(experiment_inputs, tmp_path: Path):
    corpus_root, approved_root, scene_path, base_config = experiment_inputs
    before = (approved_root / "relations_approved.jsonl").read_bytes()
    config, index, result = build_multiscale_index(
        base_config=base_config,
        enriched_scenes_path=scene_path,
        index_root=tmp_path / "multiscale_index",
        embedding_provider=FakeEmbeddingProvider(),
        corpus_root=corpus_root,
    )

    assert config.domain_id == "baseline_test_multiscale_v1"
    assert config.resolve_index_root() != base_config.resolve_index_root()
    assert result.counts == {"story": 1, "scene": 1, "card": 1, "evidence": 1}
    assert result.exact_evidence_matches == 1
    assert index.count() == 4
    evidence = next(doc for doc in index.documents if doc.document_type == "evidence")
    relation = next(doc for doc in index.documents if doc.document_type == "relation")
    assert evidence.source.line_start == 3
    assert evidence.source.line_end == 4
    assert evidence.metadata["parent_id"] == relation.id
    assert relation.metadata["scene_id"] == "scene_test_001"
    assert (approved_root / "relations_approved.jsonl").read_bytes() == before


def test_parent_aware_retrieval_and_exact_raw_output(experiment_inputs, tmp_path: Path):
    corpus_root, _, scene_path, base_config = experiment_inputs
    provider = FakeEmbeddingProvider()
    config, index, _ = build_multiscale_index(
        base_config=base_config,
        enriched_scenes_path=scene_path,
        index_root=tmp_path / "multiscale_index",
        embedding_provider=provider,
        corpus_root=corpus_root,
    )
    service = MultiScaleRagService(
        config,
        index,
        provider,
        source_extractor=OriginalTextExtractor(corpus_root),
    )

    result = service.retrieve("甲和乙是什么关系？请给出原文", top_k=4, raw_text=True)

    ids = [item["id"] for item in result["results"]]
    assert "rel_test_friend" in ids
    assert "evidence:rel_test_friend" in ids
    assert result["baseline_untouched"] is True
    assert result["context_trust"] == "untrusted_retrieved_evidence"
    evidence_result = next(item for item in result["results"] if item["id"] == "evidence:rel_test_friend")
    assert "raw_source_intent" in evidence_result["experiment_reasons"]
    assert "父场景补充" in result["context_text"]
    assert result["raw_excerpt"]["line_start"] == 3
    assert result["raw_excerpt"]["line_end"] == 4
    assert result["raw_excerpt"]["text"] == (
        "[甲] 「乙是我的旧时好友。」\n[乙] 「我们认识很多年了。」"
    )


def test_source_extractor_rejects_path_traversal(tmp_path: Path):
    extractor = OriginalTextExtractor(tmp_path)
    with pytest.raises(ValueError, match="超出语料根目录"):
        extractor.resolve_path("../outside.txt")


def test_v3_event_intent_overrides_incidental_relation_word(experiment_inputs):
    _, _, _, base_config = experiment_inputs
    analyzer = QueryAnalyzer([base_config])

    analysis = analyze_explicit_domain(analyzer, base_config.domain_id, "父亲袭击甲是怎么回事")

    assert choose_card_types(analysis, "父亲袭击甲是怎么回事") == frozenset({"fact", "event"})


def test_ab_harness_reports_both_paths_without_promotion():
    case = ABCase("relation", "甲和乙是什么关系", ("rel_test_friend",), raw_text=True)

    def baseline(_: str):
        return {"results": [{"id": "unrelated"}]}

    def experiment(_: str, *, raw_text: bool):
        assert raw_text is True
        return {"results": [{"id": "rel_test_friend"}], "raw_excerpt": {"text": "原文"}}

    result = evaluate_ab([case], baseline, experiment)
    assert result["baseline"]["recall_at_k"] == 0.0
    assert result["experiment"]["recall_at_k"] == 1.0
    assert result["experiment"]["raw_available_rate"] == 1.0
    assert result["promotion_decision"] == "manual_review_required"


def test_v2_reprocesses_each_scale_for_its_retrieval_role(experiment_inputs):
    corpus_root, _, scene_path, base_config = experiment_inputs
    result = ReprocessedMultiScaleBuilder(
        domain_id="baseline_test_multiscale_semantic_v2",
        index_version="multiscale-semantic-v2",
        aliases=base_config.aliases,
        corpus_root=corpus_root,
    ).build(base_config.source_root, scene_path)

    relation = next(doc for doc in result.documents if doc.document_type == "relation")
    evidence = next(doc for doc in result.documents if doc.document_type == "evidence")
    scene = next(doc for doc in result.documents if doc.document_type == "scene")
    assert relation.metadata["embedding_profile"] == "relation_query_aligned"
    assert "人物关系：乙是甲的朋友" in relation.embedding_text
    assert "旧时好友" in scene.embedding_text
    assert evidence.metadata["embedding_profile"] == "source_request_only"
    assert result.parented_cards == 1


def test_v2_routes_normal_questions_to_cards_and_attaches_raw_evidence(experiment_inputs, tmp_path: Path):
    corpus_root, _, scene_path, base_config = experiment_inputs
    domain_id = "baseline_test_multiscale_semantic_v2"
    build = ReprocessedMultiScaleBuilder(
        domain_id=domain_id,
        index_version="multiscale-semantic-v2",
        aliases=base_config.aliases,
        corpus_root=corpus_root,
    ).build(base_config.source_root, scene_path)
    documents = list(build.documents)
    provider = FakeEmbeddingProvider()
    vectors = provider.embed_texts([doc.embedding_text for doc in documents])
    index = DomainIndex(domain_id, tmp_path / "memory_index", provider.dimension)
    index.documents = documents
    index._rebuild_entity_index()
    index.bm25.build(documents)
    index.manifest = {"document_count": len(documents)}
    attach_vectors(index, vectors)
    config = KnowledgeDomainConfig(
        domain_id=domain_id,
        source_root=base_config.source_root,
        loader=lambda _root: documents,
        document_types=["story", "scene", "fact", "relation", "event", "evidence"],
        aliases=base_config.aliases,
        story_titles=base_config.story_titles,
    )
    service = ReprocessedMultiScaleService(
        config,
        index,
        provider,
        source_extractor=OriginalTextExtractor(corpus_root),
    )

    normal = service.retrieve("甲和乙是什么关系", top_k=5)
    assert [item["id"] for item in normal["results"]] == ["rel_test_friend"]
    assert normal["raw_excerpt"] is None

    with_source = service.retrieve("甲和乙是什么关系，请给出原文", top_k=5, raw_text=True)
    assert with_source["raw_excerpt"]["parent_id"] == "rel_test_friend"
    assert with_source["raw_excerpt"]["line_start"] == 3


def test_v3_explicit_domain_keeps_single_character_entity(experiment_inputs):
    _, _, _, base_config = experiment_inputs
    analyzer = QueryAnalyzer([base_config])
    analysis = analyze_explicit_domain(analyzer, base_config.domain_id, "甲最后怎么样了")
    assert analysis.entities == ["甲"]
    assert base_config.domain_id in analysis.matched_domains


def test_v3_routes_before_recall_and_builds_relation_timeline(experiment_inputs, tmp_path: Path):
    corpus_root, _, scene_path, base_config = experiment_inputs
    domain_id = "baseline_test_multiscale_semantic_v3"
    build = ReprocessedMultiScaleBuilder(
        domain_id=domain_id,
        index_version="multiscale-semantic-v2-1",
        aliases=base_config.aliases,
        corpus_root=corpus_root,
    ).build(base_config.source_root, scene_path)
    documents = list(build.documents)
    provider = FakeEmbeddingProvider()

    def make_index(types: frozenset[str]) -> DomainIndex:
        selected = [doc for doc in documents if doc.document_type in types]
        vectors = provider.embed_texts([doc.embedding_text for doc in selected])
        index = DomainIndex(domain_id, tmp_path / ("_".join(sorted(types))), provider.dimension)
        index.documents = selected
        index._rebuild_entity_index()
        index.bm25.build(selected)
        index.manifest = {"document_count": len(selected)}
        attach_vectors(index, vectors)
        return index

    config = KnowledgeDomainConfig(
        domain_id=domain_id,
        source_root=base_config.source_root,
        loader=lambda _root: documents,
        document_types=["story", "scene", "fact", "relation", "event", "evidence"],
        aliases=base_config.aliases,
        story_titles=base_config.story_titles,
    )
    analysis = analyze_explicit_domain(QueryAnalyzer([config]), domain_id, "甲和乙是什么关系")
    assert choose_card_types(analysis, analysis.original_query) == frozenset({"relation"})
    service = RoutedMultiScaleService(
        config,
        {
            CARD_TYPES: make_index(CARD_TYPES),
            frozenset({"relation"}): make_index(frozenset({"relation"})),
            frozenset({"story", "scene"}): make_index(frozenset({"story", "scene"})),
        },
        provider,
        all_documents=documents,
        source_extractor=OriginalTextExtractor(corpus_root),
    )
    result = service.retrieve("甲和乙是什么关系", top_k=5)
    assert result["route_types"] == ["relation"]
    assert [item["id"] for item in result["results"]] == ["rel_test_friend"]
    assert [item["id"] for item in result["relation_timeline"]] == ["rel_test_friend"]
