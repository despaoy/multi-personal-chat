import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHARACTER_DIR = PROJECT_ROOT / "backend" / "data" / "character_dialogues"
RESEARCH_DIR = CHARACTER_DIR / "experiments" / "research"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_kisaki_raw_dialogue_statistics_match_reviewed_profile():
    rows = [
        json.loads(line)
        for line in (CHARACTER_DIR / "tsukiyashiro_kisaki_raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [row["text"] for row in rows]
    assert len(texts) == 1598
    assert sum(text.count("——") for text in texts) == 161
    assert sum(len(text) <= 30 for text in texts) == 1345
    assert max(map(len, texts)) < 100


def test_kisaki_profile_facts_are_supported_and_old_absolutes_are_false():
    rows = [
        json.loads(line)
        for line in (CHARACTER_DIR / "tsukiyashiro_kisaki_raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [row["text"] for row in rows]
    assert any("亲生妹妹" in text for text in texts)
    assert any("为数不多的朋友" in text and "夜子" in text for text in texts)
    assert any("哈哈" in text for text in texts)
    assert any("嘿嘿" in text for text in texts)
    assert any("呼呼呼" in text for text in texts)
    assert not any("朋友。——因此，对她比对别人温和一些" in text for text in texts)


def test_kisaki_rag_gold_answers_are_contained_in_expected_held_out_documents():
    documents = _json(RESEARCH_DIR / "character_rag_seed_documents.json")["documents"]
    questions = _json(RESEARCH_DIR / "character_rag_retrieval_eval.json")["questions"]
    by_id = {document["id"]: document for document in documents}
    assert len(documents) == 33
    assert len(questions) == 30
    for question in questions:
        assert question["expected_doc_ids"]
        for document_id in question["expected_doc_ids"]:
            assert document_id in by_id
            assert question["gold_answer"] in by_id[document_id]["content"]
