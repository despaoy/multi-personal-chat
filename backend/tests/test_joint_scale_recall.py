"""A complete query ranks all scales without phrase-based exclusions."""

from unittest.mock import Mock

import pytest
from test_multiscale_card_recall import BODIES, arrange

from knowledge import curated_sources
from knowledge.curated_sources import validate_curated_source
from knowledge.multiscale_rag.service import RETRIEVAL_TYPES
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.retrieval_core.index import RetrievalIndexError

NEGATED = "不要完整回顾《虚构登记故事》；只核对林栖的登记身份和昨晚争执的经过。"
MIXED = "请概括《虚构登记故事》的经过，并核对林栖的登记身份及与穆远的关系。"
ORIGINAL = "虚构登记故事发生在河湾馆。\n" + "\n".join(BODIES.values())


def complete_overview(root, *, order=("scene", "fact", "relation", "event"), kind="scene"):
    rig = arrange(root, order=order)
    doc = next(doc for doc in rig.documents if doc.id == "scene")
    doc.document_type = kind
    doc.source.line_end = len(ORIGINAL.splitlines())
    (root / "scene.txt").write_text(ORIGINAL, encoding="utf-8")
    return rig


def test_negated_overview_does_not_remove_atomic_evidence(tmp_path):
    rig = complete_overview(tmp_path, order=("fact", "event", "scene", "relation"))
    result = rig.service.retrieve(NEGATED, top_k=2)
    assert {row["id"] for row in result["results"]} == {"fact", "event"}
    assert set(rig.ranking.pools[0]) == {*BODIES, "scene"}
    assert result["candidate_types"] == sorted(RETRIEVAL_TYPES)
    assert {row["excerpt"]["text"] for row in result["curated_source_catalog"]["sources"]} == {
        BODIES["fact"],
        BODIES["event"],
    }


def test_overview_without_old_trigger_phrase_can_compete_with_details(tmp_path):
    rig = complete_overview(tmp_path)
    result = rig.service.retrieve("概括《虚构登记故事》的来龙去脉。", top_k=1)
    assert result["results"][0]["id"] == "scene"
    assert result["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == ORIGINAL


def test_mixed_overview_and_atomic_tasks_keep_all_original_receipts(tmp_path, monkeypatch):
    rig = complete_overview(tmp_path)
    result = rig.service.retrieve(MIXED, top_k=3)
    assert {row["id"] for row in result["results"]} == {"scene", "fact", "relation"}
    sources = result["curated_source_catalog"]["sources"]
    assert {row["excerpt"]["text"] for row in sources} == {ORIGINAL, BODIES["fact"], BODIES["relation"]}
    monkeypatch.setattr(curated_sources, "_extractor", lambda: rig.service.extractor)
    for source in sources:
        assert validate_curated_source(source) == source


@pytest.mark.parametrize("kind", ["scene", "story"])
def test_broad_document_uses_its_own_exact_original_for_raw_output(tmp_path, kind):
    rig = complete_overview(tmp_path, kind=kind)
    result = rig.service.retrieve("请提供《虚构登记故事》的来源。", top_k=1, raw_text=True)
    source = result["curated_source_catalog"]["sources"][0]
    assert source["source_id"] == source["parent_id"] == "scene"
    assert source["excerpt"]["text"] == result["raw_excerpt"]["text"] == ORIGINAL
    assert result["raw_excerpt"]["evidence_id"] == "scene"
    assert result["raw_source_status"] == "available"


def test_unavailable_joint_vector_index_errors_without_sparse_results(tmp_path):
    rig = complete_overview(tmp_path)
    rig.service.retriever.index._faiss_index = None
    with pytest.raises(RetrievalIndexError):
        rig.service.retrieve(NEGATED)
    assert rig.ranking.pools == []


def test_declared_overview_exceeding_reader_limit_is_not_summarized(tmp_path):
    rig = complete_overview(tmp_path)
    rig.service.extractor = OriginalTextExtractor(tmp_path, max_lines=2)
    with pytest.raises(ValueError, match="max_lines"):
        rig.service.retrieve(MIXED, top_k=1)


def test_large_but_supported_complete_span_is_never_clipped(tmp_path):
    rig = complete_overview(tmp_path)
    body = "\n".join(["完整登记记录。"] * 900 + ["最后一行：不得把工作搭档解释为亲属。"])
    (tmp_path / "scene.txt").write_text(body, encoding="utf-8")
    next(doc for doc in rig.documents if doc.id == "scene").source.line_end = 901
    result = rig.service.retrieve("概括《虚构登记故事》。", top_k=1, raw_text=True)
    assert result["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == body
    assert result["raw_excerpt"]["text"] == body and not result["raw_excerpt"]["truncated"]


@pytest.mark.parametrize("error", [FileNotFoundError, UnicodeError])
def test_broad_source_read_error_never_returns_index_summary(tmp_path, monkeypatch, error):
    rig = complete_overview(tmp_path)
    failure = error("synthetic declared original unavailable")
    monkeypatch.setattr(rig.service.extractor, "extract", Mock(side_effect=failure))
    with pytest.raises(error) as caught:
        rig.service.retrieve(MIXED, top_k=1)
    assert caught.value is failure


def test_hidden_overview_original_is_never_read(tmp_path, monkeypatch):
    from knowledge.multiscale_rag.visibility import KnowledgeBoundary

    rig = complete_overview(tmp_path)
    doc = next(doc for doc in rig.documents if doc.id == "scene")
    read = Mock(side_effect=AssertionError("An unauthorized source must not be read"))
    monkeypatch.setattr(rig.service.extractor, "resolve_path", read)
    result = curated_sources.collect_curated_sources(
        [Mock(document=doc)], {}, rig.service.extractor, KnowledgeBoundary("lin", 10)
    )
    assert result == dict(sources=[], unavailable=[dict(parent_id="scene", status="no_visible_source")])
    read.assert_not_called()
