"""Cross-Encoder重排器（backend/knowledge/reranker.py）单元测试。

使用mock tokenizer/model验证（不加载真实模型）：
- 默认离线加载（local_files_only=True），联网下载需显式opt-in
- 模型缺失/加载失败时按fallback_to_original降级，且失败路径不重复尝试
- 输入candidate不被原地修改，返回副本携带rerank_score/rerank_normalized_score
- 空候选/单候选不加载模型
- GPU不可用时回退CPU且不使用CUDA专属dtype，GPU路径保持原有dtype
- 推理异常遵守fallback配置
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


# ---------- 本地加载与fallback ----------


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


def test_missing_model_returns_original_order(monkeypatch):
    _force_cpu(monkeypatch)
    _install_fake_transformers(monkeypatch, load_error=OSError("本地模型文件不存在（离线模式）"))
    candidates = _make_candidates()

    tolerant = CrossEncoderReranker(RerankConfig(model_name="missing-model", fallback_to_original=True))
    result = tolerant.rerank("查询", candidates, top_k=2)
    assert [c["id"] for c in result] == ["a", "b"]
    assert all("rerank_score" not in c for c in result)

    strict = CrossEncoderReranker(RerankConfig(model_name="missing-model", fallback_to_original=False))
    assert strict.rerank("查询", candidates, top_k=2) == []


def test_failed_path_not_retried(monkeypatch, tmp_path):
    _force_cpu(monkeypatch)
    calls = _install_fake_transformers(monkeypatch, load_error=OSError("模型缺失"))
    missing = str(tmp_path / "missing-model")  # 绝对路径：候选路径去重后仅1条

    reranker = CrossEncoderReranker(RerankConfig(model_name=missing))
    reranker.rerank("查询", _make_candidates())
    reranker.rerank("查询", _make_candidates())

    tokenizer_calls = [c for c in calls if c["kind"] == "tokenizer"]
    model_calls = [c for c in calls if c["kind"] == "model"]
    assert len(tokenizer_calls) == 1, "第二次rerank不应重复尝试加载失败的路径"
    # tokenizer加载失败后同一路径不会继续尝试model加载
    assert len(model_calls) == 0


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


def test_invalid_content_candidates_skipped(monkeypatch):
    _force_cpu(monkeypatch)
    model = _FakeModel(score_map={"甲文": 0.2, "乙文": 0.8})
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)

    reranker = CrossEncoderReranker(RerankConfig(model_name="fake-model"))
    candidates = [
        {"id": "a", "content": "甲文"},
        {"id": "empty", "content": ""},
        {"id": "non-str", "content": 123},
        {"id": "missing-content"},
        {"id": "b", "content": "乙文"},
    ]
    result = reranker.rerank("查询", candidates, top_k=5)
    assert [c["id"] for c in result] == ["b", "a"]

    # 全部无效时不加载模型，直接返回空
    all_invalid_reranker = CrossEncoderReranker(RerankConfig(model_name="fake-model"))
    assert (
        all_invalid_reranker.rerank("查询", [{"id": "x", "content": ""}, {"id": "y"}, {"id": "z", "content": None}])
        == []
    )
    assert all_invalid_reranker._model_loaded is False


def test_empty_or_single_candidates_skip_model_load(monkeypatch):
    _force_cpu(monkeypatch)
    calls = _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=_FakeModel())
    reranker = CrossEncoderReranker(RerankConfig(model_name="fake-model"))

    assert reranker.rerank("查询", []) == []
    single = [{"id": "only", "content": "唯一候选"}]
    assert reranker.rerank("查询", single) == single
    assert calls == []
    assert reranker._model_loaded is False


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


def test_inference_exception_respects_fallback(monkeypatch):
    _force_cpu(monkeypatch)
    model = _FakeModel(forward_error=RuntimeError("推理后端异常"))
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)
    candidates = _make_candidates()

    tolerant = CrossEncoderReranker(RerankConfig(model_name="fake-model", fallback_to_original=True))
    result = tolerant.rerank("查询", candidates, top_k=2)
    assert [c["id"] for c in result] == ["a", "b"]  # 原始排序

    strict = CrossEncoderReranker(RerankConfig(model_name="fake-model", fallback_to_original=False))
    assert strict.rerank("查询", candidates, top_k=2) == []


def test_inference_fallback_keeps_only_valid_candidate_copies(monkeypatch):
    _force_cpu(monkeypatch)
    model = _FakeModel(forward_error=RuntimeError("推理后端异常"))
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)
    candidates = [
        {"id": "a", "content": "甲文"},
        {"id": "invalid", "content": ""},
        {"id": "b", "content": "乙文"},
    ]

    reranker = CrossEncoderReranker(RerankConfig(model_name="fake-model", fallback_to_original=True))
    result = reranker.rerank("查询", candidates, top_k=3)

    assert [candidate["id"] for candidate in result] == ["a", "b"]
    assert result[0] is not candidates[0]
    assert result[1] is not candidates[2]


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
