"""Complete atomic sources must survive a mixed card question before ranking."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.multiscale_rag.runtime import _retrieval_index
from knowledge.multiscale_rag.service import RETRIEVAL_TYPES, RoutedMultiScaleService
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.retrieval_core.registry import KnowledgeDomainConfig
from knowledge.retrieval_scope import QueryScope

MIXED = "请核对林栖与穆远是什么关系，并说明林栖的登记身份和昨晚争执的经过。"
QUERIES = {
    "fact": "林栖的登记身份是什么？",
    "relation": "林栖与穆远是什么关系？",
    "event": "林栖与穆远昨晚争执发生了什么？",
}
BODIES = {
    "fact": "林栖登记为虚构河湾馆的档案员，受理号为十七。",
    "relation": "林栖与穆远登记为工作搭档，双方没有亲属关系。",
    "event": "昨晚林栖与穆远争执，是因为受理号复核时发现错字；双方随后协商重新填写。",
}


class ControlledEmbedding:
    def __init__(self):
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return np.ones(384, dtype=np.float32) / np.sqrt(384)


class ControlledRanking:
    def __init__(self, order):
        self.order = order
        self.pools = []

    def rerank(self, analysis, candidates, *, top_k):
        self.pools.append([candidate.document.id for candidate in candidates])
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                self.order.index(candidate.document.id) if candidate.document.id in self.order else len(self.order),
                candidate.document.id,
            ),
        )
        for index, candidate in enumerate(ranked):
            candidate.rerank_score = 0.95 - 0.05 * index
            candidate.rerank_method = "cross_encoder"
        return ranked[:top_k]


def arrange(root, *, order=("fact", "relation", "event"), empty=False):
    root.mkdir(parents=True, exist_ok=True)
    config = KnowledgeDomainConfig(
        "synthetic",
        root,
        lambda _: [],
        aliases={"林栖": "林栖", "小林": "林栖", "穆远": "穆远"},
        story_titles=["虚构登记故事"],
    )
    docs = []
    if not empty:
        for kind, body in BODIES.items():
            path = root / (kind + ".txt")
            path.write_text(body, encoding="utf-8")
            source = SourceReference(path.name, 1, 1)
            metadata = dict(subject="林栖", target="穆远", scene_id="scene", predicate="身份", relation="搭档")
            docs.append(
                KnowledgeIndexDocument(
                    kind,
                    "synthetic",
                    kind,
                    kind + "登记资料",
                    body,
                    body,
                    body,
                    entities=["林栖"] if kind == "fact" else ["林栖", "穆远"],
                    source=source,
                    metadata=metadata,
                )
            )
            docs.append(
                KnowledgeIndexDocument(
                    "source-" + kind,
                    "synthetic",
                    "evidence",
                    kind + "原文",
                    body,
                    body,
                    body,
                    source=source,
                    metadata=dict(parent_id=kind),
                )
            )
        scene_body = "虚构登记故事的场景摘要包含身份、搭档和登记争执。"
        (root / "scene.txt").write_text(scene_body, encoding="utf-8")
        docs.append(
            KnowledgeIndexDocument(
                "scene",
                "synthetic",
                "scene",
                "虚构登记故事",
                scene_body,
                scene_body,
                scene_body,
                entities=["林栖", "穆远"],
                source=SourceReference("scene.txt", 1, 1),
            )
        )
    vectors = np.ones((len(docs), 384), dtype=np.float32) / np.sqrt(384)
    index = _retrieval_index(config.domain_id, docs, vectors)
    embedding = ControlledEmbedding()
    ranking = ControlledRanking(list(order))
    service = RoutedMultiScaleService(
        config, index, embedding, all_documents=docs, source_extractor=OriginalTextExtractor(root), reranker=ranking
    )
    return SimpleNamespace(
        service=service, embedding=embedding, ranking=ranking, index=index, documents=docs, config=config
    )


@pytest.mark.parametrize("kind", ["fact", "relation", "event"])
def test_complete_atomic_sources_are_individually_retrievable(tmp_path, kind):
    rig = arrange(tmp_path, order=(kind,))
    result = rig.service.retrieve(QUERIES[kind], top_k=1)
    assert result["results"][0]["id"] == kind
    assert result["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == BODIES[kind]


def test_mixed_question_retains_every_complete_atomic_source(tmp_path):
    rig = arrange(tmp_path)
    result = rig.service.retrieve(MIXED, top_k=3)
    assert {row["id"] for row in result["results"]} == set(BODIES)
    assert set(rig.ranking.pools[0]) == {*BODIES, "scene"}
    assert {row["excerpt"]["text"] for row in result["curated_source_catalog"]["sources"]} == set(BODIES.values())
    assert result["candidate_types"] == sorted(RETRIEVAL_TYPES)


def test_existing_overview_operation_remains_retrievable(tmp_path):
    rig = arrange(tmp_path, order=("scene",))
    result = rig.service.retrieve("请完整回顾《虚构登记故事》。", top_k=1)
    assert result["results"][0]["id"] == "scene"
    assert result["candidate_types"] == sorted(RETRIEVAL_TYPES)


def test_complete_empty_card_index_is_no_match_without_model_work(tmp_path):
    rig = arrange(tmp_path, empty=True)
    rig.embedding.embed_query = Mock(side_effect=AssertionError("An empty index needs no embedding inference"))
    result = rig.service.retrieve(MIXED, top_k=3)
    assert result["results"] == [] and result["curated_source_catalog"] == dict(sources=[], unavailable=[])
    rig.embedding.embed_query.assert_not_called()


@pytest.mark.parametrize("query", ["小林是谁？", "小林是什么身份？"])
def test_identity_alias_forms_preserve_all_candidate_types(tmp_path, query):
    rig = arrange(tmp_path)
    result = rig.service.retrieve(query, top_k=3, query_scope=QueryScope(rig.config.domain_id, query, ((0, "independent_identity", ("林栖",)),)))
    assert {row["id"] for row in result["results"]} == set(BODIES)
    assert result["query_scope"]["rules"][0]["objects"] == ["林栖"]
