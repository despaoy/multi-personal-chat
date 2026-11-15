import pytest

from services.character_chat_vectors import build_memory_vectors


class Provider:
    model_id = "test"
    model_fingerprint = "weights-v1"
    dimension = 3

    def embed_texts(self, texts):
        assert texts == ["喜欢茶"]
        return self.output


@pytest.mark.parametrize("output", [[], [[1, 2]], [[0, 0, 0]], [[float('nan'), 0, 0]]])
def test_invalid_model_output_never_becomes_memory_vectors(output):
    provider = Provider()
    provider.output = output
    with pytest.raises(ValueError):
        build_memory_vectors({"memories": [{"id": "a", "content": "喜欢茶", "status": "confirmed"}]}, provider)


def test_pending_memory_is_not_embedded_and_identity_includes_weights():
    provider = Provider()
    provider.output = [[1, 0, 0]]
    identity, vectors = build_memory_vectors({"memories": [
        {"id": "a", "content": "喜欢茶", "status": "confirmed"},
        {"id": "b", "content": "未确认信息", "status": "pending"},
    ]}, provider)
    assert "weights-v1" in identity and vectors == {"a": [1, 0, 0]}
