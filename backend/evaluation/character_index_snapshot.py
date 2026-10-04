"""Copy a complete existing curated index into an isolated native evaluation."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def prepare_character_index_snapshot(source, destination, *, repository_root, runtime_root):
    repository_root = Path(repository_root).resolve(strict=True)
    runtime_root = Path(runtime_root).resolve(strict=True)
    source = Path(source)
    if source.is_symlink():
        raise ValueError("Character index source must not be a symlink")
    source = source.resolve(strict=True)
    if not any(source.is_relative_to(root) for root in (repository_root, runtime_root)):
        raise ValueError("Character index is outside authorized project roots")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or not destination.resolve().is_relative_to(runtime_root):
        raise ValueError("Character snapshot must be a new isolated runtime directory")
    files, references, counts = {}, {}, {}
    for bundle in ("card_index", "scene_story_index", "evidence_index"):
        folder = source / bundle
        if folder.is_symlink():
            raise ValueError("Character index bundle must not be a symlink")
        for filename in ("documents.jsonl", "vectors.npy", "manifest.json"):
            path = folder / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("Character index requires all complete bundle files")
            files[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        documents = [json.loads(line) for line in (folder / "documents.jsonl").read_text().splitlines() if line.strip()]
        matrix = np.load(folder / "vectors.npy", allow_pickle=False, mmap_mode="r")
        manifest = json.loads((folder / "manifest.json").read_text())
        if not documents or matrix.shape != (len(documents), 384) or manifest.get("document_count") != len(documents):
            raise ValueError("Character index manifest, documents and vectors disagree")
        counts[bundle] = len(documents)
        for document in documents:
            location = document.get("source", {}).get("source_path")
            if not isinstance(location, str) or not location:
                raise ValueError("Character index lacks original source provenance")
            original = (repository_root / location).resolve(strict=True)
            if not original.is_relative_to(repository_root) or not original.is_file():
                raise ValueError("Original character source is outside the repository")
            references[str(original)] = hashlib.sha256(original.read_bytes()).hexdigest()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, digest in files.items():
        target = destination / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        if (
            hashlib.sha256(target.read_bytes()).hexdigest() != digest
            or hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError("Character index changed during snapshot")
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in references.items()):
        raise RuntimeError("Original character source changed during snapshot")
    proof = dict(
        source_root=str(source),
        snapshot_root=str(destination.resolve()),
        complete_files=files,
        original_sources=references,
        documents_by_bundle=counts,
        original_index_unchanged=True,
        no_documents_or_sources_rewritten=True,
    )
    (destination.parent / "character-index-snapshot-private.json").write_bytes(
        (json.dumps(proof, indent=2) + chr(10)).encode()
    )
    return destination.resolve()
