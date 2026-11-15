from knowledge.retrieval_core.embedding import compute_model_fingerprint


def test_same_configuration_different_weights_or_tokenizer_invalidates_identity(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'weights-a')
    original = compute_model_fingerprint(str(tmp_path))
    weights.write_bytes(b'weights-b')
    updated = compute_model_fingerprint(str(tmp_path))
    assert updated != original
    (tmp_path / 'tokenizer.json').write_text('{"version": 2}')
    assert compute_model_fingerprint(str(tmp_path)) != updated


def test_fingerprint_is_independent_of_installation_path(tmp_path):
    for name in ('a', 'b'):
        root = tmp_path / name
        (root / '1_Pooling').mkdir(parents=True)
        (root / 'config.json').write_text('{}')
        (root / '1_Pooling/config.json').write_text('{"pooling": "mean"}')
    assert compute_model_fingerprint(str(tmp_path / 'a')) == compute_model_fingerprint(str(tmp_path / 'b'))
    (tmp_path / 'b/1_Pooling/config.json').write_text('{"pooling": "max"}')
    assert compute_model_fingerprint(str(tmp_path / 'a')) != compute_model_fingerprint(str(tmp_path / 'b'))
