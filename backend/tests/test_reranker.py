"""Cross-Encoder重排器（backend/knowledge/reranker.py）单元测试。

使用mock tokenizer/model验证（不加载真实模型）：
- 默认离线加载（local_files_only=True），联网下载需显式opt-in
- 模型缺失、加载失败和推理异常必须报错；修复后允许下一次请求重新加载
- 输入candidate不被原地修改，返回副本携带rerank_score/rerank_normalized_score
- 空候选不加载模型；单候选仍取得真实分数
- GPU不可用时回退CPU且不使用CUDA专属dtype，GPU路径保持原有dtype
- 非法输入、非有限分数不能伪装成成功
- top_k/batch_size/max_length参数边界校验
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from knowledge.reranker import CrossEncoderReranker, RerankConfig

# ---------- mock 基础设施 ----------


class _FakeBatch(dict):
    """伪造tokenizer输出：支持 .to(device) 与 model(**inputs) 展开。"""

    def __init__(self, texts):
        super().__init__(texts=list(texts))
        self.target_device = None

    def to(self, device):
        self.target_device = device
        self["device"] = device
        return self


class _FakeTokenizer:
    """伪造tokenizer：记录批量输入并返回_FakeBatch。"""

    def __init__(self):
        self.calls = []

    def __call__(self, queries, texts, truncation=True, padding=True, max_length=512, return_tensors="pt"):
        assert isinstance(queries, list) and isinstance(texts, list)
        assert len(queries) == len(texts)
        self.calls.append(list(texts))
        return _FakeBatch(texts)


class _FakeModel:
    """伪造SequenceClassification模型：按content映射打分，logits形状[B,1]。"""

    def __init__(self, score_map=None, forward_error=None):
        self.score_map = score_map or {}
        self.forward_error = forward_error
        self.target_device = None
        self.eval_called = False
        self.forward_calls = []

    def to(self, device):
        self.target_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return iter([])

    def __call__(self, **kwargs):
        texts = kwargs.get("texts", [])
        self.forward_calls.append(list(texts))
        if self.forward_error is not None:
            raise self.forward_error
        logits = torch.tensor(
            [[self.score_map.get(t, 0.0)] for t in texts],
            dtype=torch.float32,
        )
        return SimpleNamespace(logits=logits)


def _install_fake_transformers(monkeypatch, tokenizer=None, model=None, load_error=None):
    """向sys.modules注入伪造的transformers模块，返回from_pretrained调用记录。"""
    calls = []

    def _tokenizer_from_pretrained(path, **kwargs):
        calls.append({"kind": "tokenizer", "path": path, "kwargs": kwargs})
        if load_error is not None:
            raise load_error
        return tokenizer

    def _model_from_pretrained(path, **kwargs):
        calls.append({"kind": "model", "path": path, "kwargs": kwargs})
        if load_error is not None:
            raise load_error
        return model

    fake_module = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        from_pretrained = staticmethod(_tokenizer_from_pretrained)

    class _FakeAutoModelForSequenceClassification:
        from_pretrained = staticmethod(_model_from_pretrained)

    fake_module.AutoTokenizer = _FakeAutoTokenizer
    fake_module.AutoModelForSequenceClassification = _FakeAutoModelForSequenceClassification
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    return calls


def _make_candidates():
    return [
        {"id": "a", "title": "A", "content": "甲文"},
        {"id": "b", "title": "B", "content": "乙文"},
        {"id": "c", "title": "C", "content": "丙文"},
    ]


def _force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _force_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


# ---------- 本地加载与失败传播 ----------


def test_local_load_is_offline_by_default(monkeypatch):
    _force_cpu(monkeypatch)
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel())
    config = RerankConfig(model_name="fake-local-model")
    assert config.allow_download is False

    reranker = CrossEncoderReranker(config)
    result = reranker.rerank("查询", _make_candidates(), top_k=3)
    assert len(result) == 3
    assert calls, "应当调用from_pretrained加载模型"
    for call in calls:
        assert call["kwargs"]["local_files_only"] is True


def test_download_requires_explicit_opt_in(monkeypatch):
    _force_cpu(monkeypatch)
    monkeypatch.setenv("RERANKER_ALLOW_DOWNLOAD", "true")
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel())
    config = RerankConfig(model_name="fake-hub-model")
    assert config.allow_download is True

    reranker = CrossEncoderReranker(config)
    reranker.rerank("查询", _make_candidates(), top_k=3)
    assert calls
    for call in calls:
        assert call["kwargs"]["local_files_only"] is False


def test_missing_model_raises_with_original_cause(monkeypatch, caplog):
    _force_cpu(monkeypatch)
    error = OSError('private-model-load-detail')
    _install_fake_transformers(monkeypatch, load_error=error)
    reranker = CrossEncoderReranker(RerankConfig(model_name='missing-model'))
    with pytest.raises(OSError) as caught:
        reranker.rerank('查询', _make_candidates(), top_k=2)
    assert caught.value is error
    assert reranker.model is None and reranker.tokenizer is None
    assert not reranker._model_loaded
    assert 'private-model-load-detail' not in caplog.text


def test_loading_can_recover_on_next_request(monkeypatch, tmp_path):
    _force_cpu(monkeypatch)
    calls = _install_fake_transformers(monkeypatch, load_error=OSError('missing model'))
    reranker = CrossEncoderReranker(RerankConfig(model_name=str(tmp_path / 'model')))
    with pytest.raises(OSError):
        reranker.rerank('查询', _make_candidates())
    assert len(calls) == 1
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel())
    assert len(reranker.rerank('查询', _make_candidates())) == 3
    assert reranker._model_loaded


# ---------- 输入保护与输出分数 ----------


def test_candidates_copied_not_mutated(monkeypatch):
    _force_cpu(monkeypatch)
    score_map = {"甲文": 0.1, "乙文": 0.9, "丙文": 0.5}
    model = _FakeModel(score_map=score_map)
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)

    candidates = _make_candidates()
    snapshot = [dict(c) for c in candidates]
    original_ids = [id(c) for c in candidates]

    reranker = CrossEncoderReranker(RerankConfig(model_name="fake-model"))
    result = reranker.rerank("查询", candidates, top_k=3)

    # 按分数降序：乙(0.9) > 丙(0.5) > 甲(0.1)
    assert [c["id"] for c in result] == ["b", "c", "a"]
    # 原始候选未被修改
    assert candidates == snapshot
    for cand in candidates:
        assert "rerank_score" not in cand
        assert "rerank_normalized_score" not in cand
    # 返回的是副本（不同对象），原始字段保留
    assert {id(c) for c in result}.isdisjoint(set(original_ids))
    by_id = {c["id"]: c for c in result}
    assert by_id["b"]["rerank_score"] == pytest.approx(0.9)
    assert by_id["b"]["rerank_normalized_score"] == pytest.approx(1.0)
    assert by_id["c"]["rerank_normalized_score"] == pytest.approx(0.5)
    assert by_id["a"]["rerank_normalized_score"] == pytest.approx(0.0)
    assert by_id["b"]["title"] == "B"
    assert by_id["b"]["content"] == "乙文"


@pytest.mark.parametrize('invalid', [None, {'content': ''}, {'content': '  '}, {'content': 123}, {}])
def test_invalid_candidates_raise_before_loading(monkeypatch, invalid):
    _force_cpu(monkeypatch)
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel())
    reranker = CrossEncoderReranker(RerankConfig(model_name='fake-model'))
    with pytest.raises(ValueError, match='content'):
        reranker.rerank('查询', [_make_candidates()[0], invalid])
    assert calls == []


def test_empty_skips_loading_but_single_candidate_gets_model_score(monkeypatch):
    _force_cpu(monkeypatch)
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel({'唯一候选': 0.7}))
    reranker = CrossEncoderReranker(RerankConfig(model_name='fake-model'))
    assert reranker.rerank('查询', []) == []
    assert calls == []
    single = [{'id': 'only', 'content': '唯一候选'}]
    result = reranker.rerank('查询', single)
    assert result[0]['rerank_score'] == pytest.approx(0.7)
    assert result[0]['id'] == 'only' and result[0] is not single[0]
    assert 'rerank_score' not in single[0]
    assert reranker._model_loaded


# ---------- 设备处理 ----------


def test_gpu_unavailable_falls_back_to_cpu(monkeypatch):
    _force_cpu(monkeypatch)
    model = _FakeModel()
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)

    config = RerankConfig(model_name="fake-model", device="cuda:0")
    reranker = CrossEncoderReranker(config)
    assert reranker.device == "cpu"

    result = reranker.rerank("查询", _make_candidates(), top_k=3)
    assert len(result) == 3
    # CPU路径不使用CUDA专属dtype
    for call in calls:
        assert "torch_dtype" not in call["kwargs"]
    assert model.target_device == "cpu"
    assert model.eval_called is True


def test_gpu_path_keeps_existing_dtype(monkeypatch):
    _force_gpu(monkeypatch)
    model = _FakeModel()
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)

    reranker = CrossEncoderReranker(RerankConfig(model_name="fake-model", device="cuda:0"))
    assert reranker.device == "cuda:0"

    result = reranker.rerank("查询", _make_candidates(), top_k=3)
    assert len(result) == 3
    model_calls = [c for c in calls if c["kind"] == "model"]
    assert model_calls[0]["kwargs"]["torch_dtype"] == torch.float16
    assert model.target_device == "cuda:0"


@pytest.mark.parametrize('size', [1, 3])
def test_inference_error_propagates_without_partial_results(monkeypatch, size, caplog):
    _force_cpu(monkeypatch)
    error = RuntimeError('private-inference-detail')
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel(forward_error=error))
    candidates = _make_candidates()[:size]
    snapshot = [dict(item) for item in candidates]
    reranker = CrossEncoderReranker(RerankConfig(model_name='fake-model'))
    with pytest.raises(RuntimeError) as caught:
        reranker.rerank('查询', candidates)
    assert caught.value is error and candidates == snapshot
    assert 'private-inference-detail' not in caplog.text


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_model_scores_raise(monkeypatch, value):
    _force_cpu(monkeypatch)
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel({'甲文': value}))
    reranker = CrossEncoderReranker(RerankConfig(model_name='fake-model'))
    with pytest.raises(RuntimeError, match='non-finite'):
        reranker.rerank('查询', _make_candidates())


# ---------- 参数边界 ----------


@pytest.mark.parametrize("bad_top_k", [0, -3, "3", 2.5, None])
def test_invalid_top_k_raises(monkeypatch, bad_top_k):
    _force_cpu(monkeypatch)
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel())
    reranker = CrossEncoderReranker(RerankConfig(model_name="fake-model"))
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("查询", _make_candidates(), top_k=bad_top_k)


@pytest.mark.parametrize(
    "field_updates",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": 1.5},
        {"max_length": 0},
        {"max_length": -10},
    ],
)
def test_invalid_config_raises(field_updates):
    with pytest.raises(ValueError):
        RerankConfig(model_name="fake-model", **field_updates)


def test_model_adapter_and_pipeline_score_single_candidate(monkeypatch):
    from test_semantic_rag_ranking import _analysis, _candidate

    from knowledge.retrieval_core.rerank import PipelineReranker

    _force_cpu(monkeypatch)
    model = _FakeModel({'evidence': 0.8})
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)
    encoder = CrossEncoderReranker(RerankConfig(model_name='fake-model'))
    result = PipelineReranker(cross_encoder=encoder).rerank(_analysis(), [_candidate('1', 'evidence')], 1)
    assert result[0].rerank_score == pytest.approx(0.8)
    assert result[0].rerank_method == 'cross_encoder'
    assert model.forward_calls == [['evidence']]


def test_incomplete_model_scores_raise(monkeypatch):
    _force_cpu(monkeypatch)

    class IncompleteModel(_FakeModel):
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[0.2]]))

    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=IncompleteModel())
    encoder = CrossEncoderReranker(RerankConfig(model_name='fake-model'))
    with pytest.raises(RuntimeError, match='incomplete'):
        encoder.rerank('查询', _make_candidates())


@pytest.mark.parametrize('fail', [False, True])
def test_requested_warmup_is_completed_or_raises(monkeypatch, fail):
    _force_cpu(monkeypatch)
    error = OSError('warmup failure') if fail else None

    class WarmupTokenizer(_FakeTokenizer):
        def __call__(self, queries, texts, **kwargs):
            if isinstance(queries, str):
                queries, texts = [queries], [texts]
            return super().__call__(queries, texts, **kwargs)

    _install_fake_transformers(monkeypatch, tokenizer=WarmupTokenizer(), model=_FakeModel(forward_error=error))
    encoder = CrossEncoderReranker(RerankConfig(model_name='fake-model', warmup_on_init=True))
    if fail:
        with pytest.raises(OSError) as caught:
            encoder.rerank('查询', _make_candidates())
        assert caught.value is error
        assert not encoder._model_loaded and not encoder._warmup_done
        assert encoder.model is None and encoder.tokenizer is None
    else:
        assert len(encoder.rerank('查询', _make_candidates())) == 3
        assert encoder._model_loaded and encoder._warmup_done


def test_config_reads_model_path_when_constructed(monkeypatch):
    monkeypatch.setenv('RERANKER_MODEL_PATH', '/models/first')
    first = RerankConfig()
    monkeypatch.setenv('RERANKER_MODEL_PATH', '/models/second')
    assert first.model_name == '/models/first'
    assert RerankConfig().model_name == '/models/second'
    assert RerankConfig(model_name='/models/explicit').model_name == '/models/explicit'


def test_failed_relative_model_load_never_tries_another_directory(monkeypatch):
    _force_cpu(monkeypatch)
    error = OSError('configured model missing')
    calls = _install_fake_transformers(monkeypatch, load_error=error)
    encoder = CrossEncoderReranker(RerankConfig(model_name='relative/model'))
    with pytest.raises(OSError) as caught:
        encoder.rerank('查询', _make_candidates())
    assert caught.value is error
    assert len(calls) == 1 and calls[0]['path'] == 'relative/model'
    assert calls[0]['kwargs']['local_files_only'] is True
    assert encoder.model is None and encoder.tokenizer is None


def test_explicit_model_path_reaches_tokenizer_and_model_unchanged(monkeypatch):
    _force_cpu(monkeypatch)
    monkeypatch.setenv('RERANKER_MODEL_PATH', '/models/environment')
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel())
    encoder = CrossEncoderReranker(RerankConfig(model_name='chosen/model'))
    assert len(encoder.rerank('查询', _make_candidates())) == 3
    assert [call['path'] for call in calls] == ['chosen/model', 'chosen/model']
