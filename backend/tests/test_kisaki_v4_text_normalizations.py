from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = (
    ROOT
    / "backend/data/character_dialogues/experiments/v4/text_normalizations.json"
)
TRAIN = ROOT / "backend/data/character_dialogues/experiments/v4/train.jsonl"
RAW = ROOT / "backend/data/character_dialogues/tsukiyashiro_kisaki_raw.jsonl"
RAG = (
    ROOT
    / "backend/data/character_dialogues/experiments/research/character_rag_seed_documents.json"
)


def test_surface_normalizations_are_applied_without_rewriting_raw_sources() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    train = [
        json.loads(line)
        for line in TRAIN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_by_id = {row["id"]: row for row in train}
    raw_text = RAW.read_text(encoding="utf-8")
    rag = json.loads(RAG.read_text(encoding="utf-8"))
    rag_by_id = {document["id"]: document for document in rag["documents"]}

    assert ledger["policy"] == "surface_only_no_semantic_edit"
    assert ledger["raw_sources_modified"] is False
    assert len(ledger["items"]) == 3

    for item in ledger["items"]:
        assert item["original"] in raw_text
        if "canonical_train" in item["targets"]:
            assistant_texts = [
                message["content"]
                for message in train_by_id[item["record_id"]]["messages"]
                if message["role"] == "assistant"
            ]
            assert item["normalized"] in assistant_texts
            assert item["original"] not in assistant_texts
        if "rag_seed_documents" in item["targets"]:
            content = rag_by_id[item["document_id"]]["content"]
            assert item["normalized"] in content
            assert item["original"] not in content


def test_surface_normalizations_do_not_change_gold_prompts() -> None:
    gold = json.loads(
        (ROOT / "backend/evaluation/kisaki_gold_set_v3.json").read_text(
            encoding="utf-8"
        )
    )
    prompt = next(row for row in gold["prompts"] if row["id"] == "kisaki_v3_rag_021")

    assert prompt["prompt"] == "证据都涉及学校，哪段明确表达妃期待明年？"
