"""Audit Gold v2 candidates against canonical data using sentence embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import dialogue_prompts  # noqa: E402

EXPERIMENT_DIR = BACKEND / "data" / "character_dialogues" / "experiments"
DEFAULT_CANDIDATES = BACKEND / "evaluation" / "kisaki_gold_set_v2_candidates.json"
DEFAULT_AUDIT = EXPERIMENT_DIR / "gold_v2_leakage_audit.json"
_LAB_ROOT = os.getenv("QQCHAT_LAB_ROOT", "/home/szw/lhm2")
DEFAULT_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    str(Path(_LAB_ROOT) / "runtime" / "models" / "bge-small-zh-v1.5"),
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic leakage audit for Kisaki Gold v2")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.88)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers is required for semantic audit") from exc

    candidate_data = _load(args.candidates)
    train = _load(EXPERIMENT_DIR / "tsukiyashiro_kisaki_train.json")
    validation = _load(EXPERIMENT_DIR / "tsukiyashiro_kisaki_eval.json")
    candidate_rows: list[tuple[str, str]] = []
    for item in candidate_data.get("prompts", []):
        turns = item.get("turns") or [item.get("prompt", "")]
        candidate_rows.extend((f"{item['id']}#turn-{index}", str(turn)) for index, turn in enumerate(turns))
    reference_rows = [
        (f"{item.get('id')}#turn-{index}", prompt)
        for item in train + validation
        for index, prompt in enumerate(dialogue_prompts(item))
    ]

    model = SentenceTransformer(args.model)
    candidate_vectors = model.encode(
        [text for _, text in candidate_rows],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    reference_vectors = model.encode(
        [text for _, text in reference_rows],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = candidate_vectors @ reference_vectors.T
    previous = _load(args.audit) if args.audit.exists() else {}
    resolutions = {
        item.get("candidate_id"): item.get("resolution")
        for item in previous.get("semantic_matches", [])
        if item.get("resolution")
    }
    matches: list[dict[str, Any]] = []
    for index, (candidate_id, candidate_text) in enumerate(candidate_rows):
        best_index = int(scores[index].argmax())
        best_score = float(scores[index][best_index])
        if best_score >= args.threshold:
            reference_id, reference_text = reference_rows[best_index]
            matches.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_prompt": candidate_text,
                    "reference_id": reference_id,
                    "reference_prompt": reference_text,
                    "similarity": round(best_score, 6),
                    "resolution": resolutions.get(candidate_id, ""),
                }
            )
    unresolved = [item for item in matches if item.get("resolution") not in {"accepted_distinct", "candidate_rewritten"}]
    audit = {
        **previous,
        "semantic_similarity_threshold": args.threshold,
        "semantic_model": args.model,
        "semantic_audit_status": "passed" if not unresolved else "review_required",
        "semantic_matches": matches,
        "unresolved_semantic_matches": unresolved,
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "semantic_audit_status": audit["semantic_audit_status"],
                "matches": len(matches),
                "unresolved": len(unresolved),
                "audit": str(args.audit),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
