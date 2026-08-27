#!/usr/bin/env python3
"""P4E: approve reviewed scene metadata and write enriched scene artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from knowledge.game_rag import (  # noqa: E402
    ReviewStatus,
    SceneMetadataReviewDocument,
    load_frozen_scene_bundle,
    load_scene_metadata_review,
    save_scene_metadata_review,
    validate_scene_metadata_review,
    write_enriched_scenes,
)

BASE_DIR = BACKEND_DIR / "data" / "knowledge" / "tsukiyashiro_kisaki"
BOUNDARY_DIR = BASE_DIR / "scene_boundary_review"
REVIEW_DIR = BASE_DIR / "scene_metadata_review"
OUTPUT_DIR = BASE_DIR / "scene_metadata_enriched"


def _approved_document(review: SceneMetadataReviewDocument) -> SceneMetadataReviewDocument:
    decisions = [
        decision.model_copy(update={"review_status": ReviewStatus.approved}, deep=True)
        for decision in review.scene_decisions
    ]
    return SceneMetadataReviewDocument.model_validate(
        review.model_copy(
            update={"review_status": "approved", "scene_decisions": decisions},
            deep=True,
        ).model_dump(mode="json")
    )


def main() -> None:
    bundle = load_frozen_scene_bundle(
        BOUNDARY_DIR / "scenes.jsonl",
        BOUNDARY_DIR / "boundary_manifest.json",
    )
    review_path = REVIEW_DIR / "scene_metadata_review.json"
    review = load_scene_metadata_review(review_path)
    errors = validate_scene_metadata_review(review, bundle, require_complete=True)
    if errors:
        raise ValueError("P4D review is not complete:\n- " + "\n- ".join(errors))

    if review.review_status == "approved":
        approved = review
    else:
        invalid = [
            decision.scene_id
            for decision in review.scene_decisions
            if decision.review_status is not ReviewStatus.needs_review
        ]
        if invalid:
            raise ValueError(f"P4D records must all be needs_review before approval: {invalid[:5]}")
        approved = _approved_document(review)
        save_scene_metadata_review(review_path, approved)

    manifest = write_enriched_scenes(bundle, approved, OUTPUT_DIR)
    print(f"[P4E] approved and enriched: scenes={manifest['total_scenes']} output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
