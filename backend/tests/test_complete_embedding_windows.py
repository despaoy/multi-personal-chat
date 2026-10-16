"""Complete token coverage and explicit errors, using a declared tiny tokenizer."""

from unittest.mock import Mock

import numpy as np
import pytest

from knowledge.retrieval_core import embedding as windows


class Tokenizer:
    def __init__(self):
        self.payloads = []
        self.calls = 0

    def num_special_tokens_to_add(self, pair):
        assert pair is False
        return 2

    def __call__(self, texts, **kwargs):
        self.calls += 1
        tokens = [list(map(int, text.split())) for text in texts]
        if kwargs == dict(add_special_tokens=False, truncation=False, padding=False):
            return {"input_ids": tokens}
        assert kwargs == dict(truncation=True, max_length=5, stride=0, return_overflowing_tokens=True,
                              return_special_tokens_mask=True, padding=True, return_tensors="pt")
        rows, owners = [], []
        for owner, row in enumerate(tokens):
            for start in range(0, max(1, len(row)), 3):
                payload = row[start:start + 3]
                self.payloads.append(payload)
                rows.append([100, *payload, 101])
                owners.append(owner)
        return dict(input_ids=np.asarray([row + [0] * (5-len(row)) for row in rows]),
                    attention_mask=np.asarray([[1] * len(row) + [0] * (5-len(row)) for row in rows]),
                    special_tokens_mask=np.asarray([[1] + [0] * (len(row)-2) + [1] + [1] * (5-len(row)) for row in rows]),
                    overflow_to_sample_mapping=np.asarray(owners))


def encode(rows):
    payloads = [[token for token, active in zip(row, mask, strict=True) if active][1:-1]
                for row, mask in zip(rows["input_ids"], rows["attention_mask"], strict=True)]
    return np.asarray([[sum(payload)+1, len(payload)] for payload in payloads], dtype=np.float32)


def run(texts, tokenizer, **kwargs):
    options = dict(max_tokens=5, batch_size=2, dimension=2, encode_batch=encode)
    options.update(kwargs)
    return windows.complete_embeddings(texts, tokenizer, **options)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_every_payload_token_is_encoded_once_without_mixing_text_owners(batch_size):
    tokenizer = Tokenizer()
    result = run(["1 2 3 4 5 6 7", "8 9"], tokenizer, batch_size=batch_size)
    assert tokenizer.payloads == [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
    raw = np.asarray([[sum(payload)+1, len(payload)] for payload in tokenizer.payloads], dtype=np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    expected = np.asarray([3 * raw[0] + 3 * raw[1] + raw[2], raw[3]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.shape == (2, 2) and result.dtype == np.float32


@pytest.mark.parametrize("length,counts", [(3, [3]), (4, [3, 1])])
def test_exact_payload_boundary_and_one_extra_token_are_complete(length, counts):
    tokenizer = Tokenizer()
    run([" ".join(str(i + 1) for i in range(length))], tokenizer)
    assert [len(payload) for payload in tokenizer.payloads] == counts


def test_empty_text_and_empty_collection_have_explicit_representations():
    tokenizer = Tokenizer()
    np.testing.assert_allclose(run([""], tokenizer), [[1, 0]])
    empty = Tokenizer()
    assert run([], empty).shape == (0, 2) and empty.calls == 0


@pytest.mark.parametrize("kwargs", [dict(batch_size=0), dict(max_tokens=True), dict(max_tokens=2),
    dict(dimension=0), dict(timeout_seconds=0), dict(timeout_seconds=float("nan"))])
def test_invalid_profile_cannot_be_clamped_or_encoded(kwargs):
    model = Mock(side_effect=AssertionError("Invalid profile must stop before inference"))
    with pytest.raises(ValueError):
        run(["1"], Tokenizer(), encode_batch=model, **kwargs)
    model.assert_not_called()


@pytest.mark.parametrize("failure_at", [1, 2])
def test_failed_inference_never_returns_or_repeats_a_partial_embedding(failure_at):
    error = RuntimeError("controlled model failure")
    count = 0

    def fail(rows):
        nonlocal count
        count += 1
        if count == failure_at:
            raise error
        return encode(rows)

    with pytest.raises(RuntimeError) as caught:
        run(["1 2 3 4 5 6 7"], Tokenizer(), batch_size=1, encode_batch=fail)
    assert caught.value is error and count == failure_at


@pytest.mark.parametrize("bad", [np.ones((2, 3)), np.zeros((1, 2)), np.asarray([[float("nan"), 1]])])
def test_invalid_vector_output_is_not_averaged_into_success(bad):
    with pytest.raises(ValueError):
        run(["1"], Tokenizer(), encode_batch=lambda rows: bad)


def test_timeout_stops_before_the_next_batch(monkeypatch):
    clock = iter([0, 0, 2])
    monkeypatch.setattr(windows.time, "monotonic", lambda: next(clock))
    model = Mock(side_effect=encode)
    with pytest.raises(TimeoutError):
        run(["1 2 3 4 5 6 7"], Tokenizer(), batch_size=1, encode_batch=model, timeout_seconds=1)
    assert model.call_count == 1


def test_tokenizer_row_loss_is_an_error_before_inference():
    class MissingRows(Tokenizer):
        def __call__(self, *args, **kwargs):
            return {"input_ids": []}

    with pytest.raises(ValueError, match="number of source texts"):
        run(["1"], MissingRows())


def test_missing_overflow_window_fails_before_model_inference():
    class IncompleteWindows(Tokenizer):
        def __call__(self, texts, **kwargs):
            result = super().__call__(texts, **kwargs)
            if kwargs.get("return_overflowing_tokens"):
                return {name: value[:1] for name, value in result.items()}
            return result

    model = Mock(side_effect=AssertionError("Missing tokens must stop before inference"))
    with pytest.raises(ValueError, match="complete source tokens"):
        run(["1 2 3 4"], IncompleteWindows(), encode_batch=model)
    model.assert_not_called()


@pytest.mark.parametrize("caller", ["shared", "character"])
@pytest.mark.parametrize("batch_size", [0, True, 1.5])
def test_provider_configuration_is_not_silently_clamped(caller, batch_size):
    if caller == "shared":
        provider = windows.SentenceTransformerEmbeddingProvider(model_path="synthetic", batch_size=batch_size)
        model = Mock()
        model.tokenizer = Tokenizer()
        model.max_seq_length = 5
        provider._model = model
    else:
        from knowledge.multiscale_rag.vector_runtime import LocalMeanPoolingEmbeddingProvider

        provider = LocalMeanPoolingEmbeddingProvider(model_path="synthetic", batch_size=batch_size, max_length=5)
        provider._model = Mock()
        provider._tokenizer = Tokenizer()
    with pytest.raises(ValueError, match="positive integers"):
        provider.embed_texts(["1"])
