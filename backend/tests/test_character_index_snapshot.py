"""An available-character positive requires whole unmodified original inputs."""

import hashlib
import json

import numpy as np
import pytest
from evaluation.character_index_snapshot import prepare_character_index_snapshot


def setup(tmp_path):
    repo = tmp_path / "repository"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    runtime.mkdir()
    source = repo / "curated"
    source.mkdir()
    original = repo / "complete-fictional-source.txt"
    original.write_text("独立合成关系资料，前项限制与末尾否定均完整。" + "完整原始证据。" * 160)
    for kind in ["card_index", "scene_story_index", "evidence_index"]:
        folder = source / kind
        folder.mkdir()
        row = dict(id="fictional:" + kind, source=dict(source_path=original.name), content=original.read_text())
        (folder / "documents.jsonl").write_text(json.dumps(row) + "\n")
        np.save(folder / "vectors.npy", np.zeros((1, 384), dtype=np.float32))
        (folder / "manifest.json").write_text(json.dumps(dict(document_count=1)))
    return repo, runtime, source, original


def test_complete_snapshot_preserves_every_original_byte_and_provenance(tmp_path):
    repo, runtime, source, original = setup(tmp_path)
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    body = original.read_bytes()
    destination = prepare_character_index_snapshot(
        source, runtime / "evaluation" / "character-index", repository_root=repo, runtime_root=runtime
    )
    assert all(
        (destination / name).read_bytes() == data == (source / name).read_bytes() for name, data in before.items()
    )
    assert original.read_bytes() == body
    proof = json.loads((destination.parent / "character-index-snapshot-private.json").read_text())
    assert (
        len(proof["complete_files"]) == 9
        and len(proof["original_sources"]) == 1
        and proof["original_sources"][str(original.resolve())] == hashlib.sha256(body).hexdigest()
    )


@pytest.mark.parametrize(
    "defect",
    ["missing_bundle", "shape", "manifest", "missing_original", "outside_original", "existing_target", "outside_index"],
)
def test_incomplete_or_out_of_scope_snapshot_rejected_before_cloud_or_database_work(tmp_path, defect):
    repo, runtime, source, original = setup(tmp_path)
    destination = runtime / "evaluation" / "character-index"
    if defect == "missing_bundle":
        (source / "evidence_index" / "vectors.npy").unlink()
    elif defect == "shape":
        np.save(source / "card_index" / "vectors.npy", np.zeros((1, 12), dtype=np.float32))
    elif defect == "manifest":
        (source / "card_index" / "manifest.json").write_text(json.dumps(dict(document_count=2)))
    elif defect == "missing_original":
        original.unlink()
    elif defect == "outside_original":
        outside = tmp_path / "outside.txt"
        outside.write_text("完整但不属于授权索引作用域")
        row = json.loads((source / "card_index" / "documents.jsonl").read_text())
        row["source"]["source_path"] = str(outside)
        (source / "card_index" / "documents.jsonl").write_text(json.dumps(row) + "\n")
    elif defect == "existing_target":
        destination.mkdir(parents=True)
    else:
        source = tmp_path / "outside-index"
        source.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_character_index_snapshot(source, destination, repository_root=repo, runtime_root=runtime)
    assert not (destination.parent / "character-index-snapshot-private.json").exists()


def test_source_change_during_copy_is_not_certified_as_a_complete_snapshot(tmp_path, monkeypatch):
    from evaluation import character_index_snapshot as module

    repo, runtime, source, _ = setup(tmp_path)
    copy = module.shutil.copyfile

    def changed(before, after):
        result = copy(before, after)
        if before.name == "documents.jsonl":
            before.write_bytes(before.read_bytes() + b" ")
        return result

    monkeypatch.setattr(module.shutil, "copyfile", changed)
    with pytest.raises(RuntimeError, match="changed during snapshot"):
        prepare_character_index_snapshot(
            source, runtime / "evaluation" / "character-index", repository_root=repo, runtime_root=runtime
        )
