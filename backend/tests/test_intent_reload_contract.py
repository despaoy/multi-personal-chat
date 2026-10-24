"""Isolated real classifier training; only external encoder weights are synthetic.

Full reviewed samples, sklearn/CV/scaler/joblib, API admission, background job,
status route and classifier prediction stay real. No language model or network.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from api import knowledge
from app.dependencies import get_current_admin
from knowledge import intent_detector as detector
from knowledge import intent_trainer as trainer

if TYPE_CHECKING:
    from pathlib import Path


LABEL = "虚构青屿档案馆"
PUBLIC = (
    "核对虚构青屿档案馆周二十点至十六点的开放时间，周一不开放。",
    "查询虚构青屿档案馆普通票三十七元，未满六岁免费但须预约。",
    "核对虚构青屿档案馆雨天是否仅关闭室外展区，室内照常开放。",
    "查询虚构青屿档案馆预约须于前一天十八点前提交，不能当天补办。",
    "核对虚构青屿档案馆周六只接受已预约观众，持票也不能免预约。",
    "查询虚构青屿档案馆团体至少十人，节假日不提供团体折扣。",
)
NEGATIVE = (
    "今天我只想说说心情，不需要任何公开资料。",
    "谢谢你刚才听我倾诉，这条仅表达感谢，不用查询知识库。",
    "我现在想安静待一会儿，不想讨论档案馆或索取事实。",
    "这只是今天的私人感受，不是对青屿档案馆的查询。",
    "晚安，今天的对话就到这里，无须继续检索。",
    "我在记录自己的情绪，没有要求查开放时间、费用或预约规则。",
)
NEGATIVE_QUERY = "解释一下这只是我的心情记录；不查询外部，我现在只想安静聊天。"
PRIVATE_FORM_QUERY = "我喜欢什么？"


class SyntheticWeightsError(RuntimeError):
    pass


@dataclass
class TrainingRig:
    model_dir: Path
    source: Path
    constructor_calls: list[tuple[str, int]] = field(default_factory=list)
    encoded: list[tuple[str, ...]] = field(default_factory=list)
    fail_after: int | None = None

    def constructor(self, source, **_kwargs):
        source = str(source)
        self.constructor_calls.append((source, threading.get_ident()))
        if source != str(self.source) or (
            self.fail_after is not None and len(self.constructor_calls) > self.fail_after
        ):
            raise SyntheticWeightsError("isolated encoder construction failed")
        rig = self

        class Encoder:
            def __init__(self):
                self.source = source

            def get_sentence_embedding_dimension(self):
                return 4

            def encode(self, texts, **_kwargs):
                texts = tuple(texts)
                rig.encoded.append(texts)
                # Explicit test vectors replace only pretrained weights. These
                # are not language rules, inferred labels or model-quality proof.
                vectors = {text: (4.0, 0.0, 1.0, 0.0) for text in PUBLIC}
                vectors.update({text: (0.0, 4.0, 0.0, 1.0) for text in NEGATIVE})
                vectors[NEGATIVE_QUERY] = (0.0, 4.0, 0.0, 1.0)
                vectors[PRIVATE_FORM_QUERY] = (4.0, 0.0, 1.0, 0.0)
                # Old reload called an arbitrary query; keep that query complete
                # so baseline diagnoses false completion rather than test data.
                vectors["测试"] = (0.0, 4.0, 0.0, 1.0)
                return np.asarray([vectors[text] for text in texts], dtype=float)

        return Encoder()


def arrange_intent_training(monkeypatch, tmp_path):
    """Public arrange helper for the root's isolated baseline preflight."""
    source = (tmp_path / "synthetic-local-encoder").resolve()
    source.mkdir()
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    backend_dir = tmp_path / "isolated_backend"
    backend_dir.mkdir()
    rig = TrainingRig((tmp_path / "classifier").resolve(), source)
    monkeypatch.setattr(trainer, "BACKEND_DIR", backend_dir)
    monkeypatch.setattr(trainer, "MODEL_DIR", rig.model_dir)
    monkeypatch.setattr(trainer, "SAMPLES_DIR", samples_dir)
    monkeypatch.setenv("INTENT_MODEL_PATH", str(rig.model_dir))
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(source))
    monkeypatch.delenv("EMBED_MODEL_PATH", raising=False)
    monkeypatch.setattr(trainer, "_training_status", {"running": False})
    monkeypatch.setattr(trainer, "_generation_status", {"running": False})
    monkeypatch.setattr(trainer, "_training_logs", [])
    monkeypatch.setattr(trainer, "_cancel_flag", threading.Event())
    for name, value in (
        ("_ML_AVAILABLE", False), ("_ml_encoder", None),
        ("_ml_classifier", None), ("_ml_config", None),
    ):
        monkeypatch.setattr(detector, name, value)
    for name in ("_intent_task", "_intent_task_lock", "_intent_task_lock_loop"):
        monkeypatch.setattr(knowledge, name, None)
    weights = ModuleType("sentence_transformers")
    weights.SentenceTransformer = rig.constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", weights)
    trainer.save_samples({LABEL: list(PUBLIC), "none": list(NEGATIVE)})
    return rig


@pytest.fixture
def training_rig(monkeypatch, tmp_path):
    return arrange_intent_training(monkeypatch, tmp_path)


async def run_training_http():
    """Exercise real HTTP admission, owned task and the public status route."""
    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin", "user_id": 901}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture.invalid") as client:
        response = await client.post("/api/knowledge/train-intent", json={})
        assert response.status_code == 200 and response.json()["success"] is True
        task = knowledge._intent_task
        if task is not None:
            await asyncio.wait_for(task, timeout=30)
        response = await client.get("/api/knowledge/train-intent/status")
        assert response.status_code == 200 and response.json()["success"] is True
        status = response.json()["status"]
        assert status["running"] is False
        return status


def assert_unavailable():
    assert detector._ML_AVAILABLE is False
    assert detector._ml_encoder is detector._ml_classifier is detector._ml_config is None


def saved_files(rig):
    return all((rig.model_dir / name).is_file() for name in ("config.json", "classifier.joblib", "scaler.joblib"))


async def baseline_preflight(monkeypatch, tmp_path):
    """Safe old/new summary; zero paid calls, unchanged production files/data."""
    rig = arrange_intent_training(monkeypatch, tmp_path)
    rig.fail_after = 1  # Training encoder works; every reload constructor fails.
    status = await run_training_http()
    result = status["result"]
    return {
        "artifacts_saved": saved_files(rig),
        "stage": status["stage"],
        "reported_success": result.get("success") is True,
        "reported_error": bool(result.get("error")),
        "reported_partial": result.get("artifacts_saved") is True and result.get("reloaded") is False,
        "classifier_available": detector._ML_AVAILABLE,
        "encoder_constructor_attempts": len(rig.constructor_calls),
        "synthetic_encoder_weights": True,
        "paid_model_calls": 0,
    }


async def test_http_training_uses_one_real_encoder_source_and_publishes_complete_model(training_rig):
    rig = training_rig
    event_loop_thread = threading.get_ident()
    status = await run_training_http()
    result = status["result"]
    assert status["stage"] == "done" and result["success"] is True and result["reloaded"] is True
    assert result["model_path"] == str(rig.model_dir) and result["training_samples"] == 12
    assert result["accuracy"] == 1.0 and saved_files(rig)
    config = json.loads((rig.model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["embedding_model"] == str(rig.source)
    assert config["label_names"] == [LABEL, "none"]
    assert config["cv_accuracy_mean"] == 1.0
    assert detector._ML_AVAILABLE is True
    assert isinstance(detector._ml_classifier, LogisticRegression)
    assert isinstance(detector._ml_config["_scaler"], StandardScaler)
    assert detector._ml_encoder.source == str(rig.source)
    assert [source for source, _thread in rig.constructor_calls] == [str(rig.source)] * 2
    assert all(thread != event_loop_thread for _source, thread in rig.constructor_calls)
    assert set(rig.encoded[0]) == set(PUBLIC + NEGATIVE)


@pytest.mark.parametrize("missing", ["config.json", "scaler.joblib"])
def test_explicit_reload_missing_artifact_clears_old_model_without_using_ambient_directory(
    training_rig, monkeypatch, tmp_path, missing,
):
    rig = training_rig
    trainer._train_multiclass_model({LABEL: list(PUBLIC)}, list(NEGATIVE))
    detector._load_ml_model(rig.model_dir)
    assert detector._ML_AVAILABLE is True
    ambient = tmp_path / "other-complete-artifact"
    shutil.copytree(rig.model_dir, ambient)
    monkeypatch.setenv("INTENT_MODEL_PATH", str(ambient))
    (rig.model_dir / missing).unlink()
    attempts = len(rig.constructor_calls)
    with pytest.raises(FileNotFoundError):
        detector._load_ml_model(rig.model_dir)
    assert_unavailable()
    assert len(rig.constructor_calls) == attempts
    assert (ambient / missing).is_file()


def test_encoder_constructor_failure_cannot_publish_partial_classifier(training_rig):
    rig = training_rig
    trainer._train_multiclass_model({LABEL: list(PUBLIC)}, list(NEGATIVE))
    detector._load_ml_model(rig.model_dir)
    rig.fail_after = len(rig.constructor_calls)
    with pytest.raises(SyntheticWeightsError, match="isolated encoder"):
        detector._load_ml_model(rig.model_dir)
    assert_unavailable()
    assert saved_files(rig)
    assert [source for source, _thread in rig.constructor_calls] == [str(rig.source)] * 3


async def test_saved_artifacts_reload_failure_is_partial_error_in_public_status(training_rig):
    rig = training_rig
    rig.fail_after = 1
    status = await run_training_http()
    result = status["result"]
    assert status["stage"] == "error" and status["progress"] != 100
    assert result["success"] is False and result["artifacts_saved"] is True and result["reloaded"] is False
    assert result["model_path"] == str(rig.model_dir) and result["accuracy"] == 1.0
    assert "isolated encoder" in result["error"] and saved_files(rig)
    assert not any("意图检测器已重载" in line for line in status["logs"])
    assert_unavailable()
    assert [source for source, _thread in rig.constructor_calls] == [str(rig.source)] * 2


async def test_cross_validation_failure_does_not_save_or_report_zero_metric_success(training_rig, monkeypatch):
    from sklearn import model_selection

    rig = training_rig

    def broken_cv(estimator, features, labels, **kwargs):
        assert isinstance(estimator, LogisticRegression) and features.shape == (12, 4)
        assert np.unique(labels).tolist() == [0, 1] and np.bincount(labels).tolist() == [6, 6]
        assert kwargs == {"cv": 2, "scoring": "accuracy", "error_score": "raise"}
        raise ValueError("isolated CV evaluation failed with complete samples")

    monkeypatch.setattr(model_selection, "cross_val_score", broken_cv)
    status = await run_training_http()
    assert status["stage"] == "error"
    assert "isolated CV" in status["result"]["error"]
    assert status["result"].get("success") is not True
    assert not any((rig.model_dir / name).exists() for name in ("config.json", "classifier.joblib", "scaler.joblib"))
    assert_unavailable()


@pytest.mark.parametrize("query, expected", [(NEGATIVE_QUERY, False), (PRIVATE_FORM_QUERY, True)])
def test_classifier_decisions_are_not_overridden_by_rules_or_private_keyword_shortcuts(
    training_rig, monkeypatch, query, expected,
):
    rig = training_rig
    trainer._train_multiclass_model({LABEL: list(PUBLIC)}, list(NEGATIVE))
    detector._load_ml_model(rig.model_dir)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Classifier decision must not call the legacy rule engine or private shortcut")

    from knowledge import task_dependency

    monkeypatch.setattr(task_dependency, "local_context_only", forbidden)
    before = len(rig.encoded)
    result, confidence, selected = detector.predict_rag_intent(query)
    assert result is expected and 0.0 <= confidence <= 1.0
    assert selected == (LABEL if expected else None)
    assert len(rig.encoded) == before + 1 and rig.encoded[-1] == (query,)


async def test_cancelled_background_reload_cannot_publish_when_reader_thread_finishes(
    training_rig, monkeypatch,
):
    rig = training_rig
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    actual_read = detector._read_ml_model

    def blocked_read(model_dir):
        entered.set()
        try:
            if not release.wait(10):
                raise TimeoutError("isolated reader was not released")
            # Finish the actual artifact/encoder read after the owner cancelled.
            # The worker returns real artifacts but owns no global publication.
            return actual_read(model_dir)
        finally:
            finished.set()

    monkeypatch.setattr(detector, "_read_ml_model", blocked_read)
    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_admin] = lambda: {"role": "admin", "user_id": 901}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture.invalid") as client:
        try:
            response = await client.post("/api/knowledge/train-intent", json={})
            assert response.status_code == 200 and response.json()["success"] is True
            task = knowledge._intent_task
            assert task is not None
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=6)
            assert trainer.get_training_status()["stage"] == "reload" and saved_files(rig)
            assert_unavailable()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            response = await client.get("/api/knowledge/train-intent/status")
            assert response.status_code == 200
            cancelled_status = response.json()["status"]
            assert cancelled_status["running"] is False and cancelled_status["stage"] == "cancelled"
            assert cancelled_status["result"]["cancelled"] is True
            assert_unavailable()
        finally:
            release.set()
            if entered.is_set():
                assert await asyncio.wait_for(asyncio.to_thread(finished.wait, 5), timeout=6)
        await asyncio.sleep(0)  # Deliver the completed executor future callback.
        response = await client.get("/api/knowledge/train-intent/status")
        assert response.status_code == 200 and response.json()["status"] == cancelled_status
    assert [source for source, _thread in rig.constructor_calls] == [str(rig.source)] * 2
    assert not any("意图检测器已重载" in line for line in trainer.get_training_status()["logs"])
    assert_unavailable()
