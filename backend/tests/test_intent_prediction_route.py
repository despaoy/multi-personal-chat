"""Optional intent classifiers may decide routes, but failed predictions cannot."""


import pytest

from knowledge import intent_detector

QUERY = "核对虚构夏屿展览的费用、开放时段与雨天例外；无对应来源则保持未知。"


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
def test_loaded_classifier_failure_is_not_replaced_by_rules(monkeypatch, error_type):
    error = error_type("Declared isolated prediction failure")

    def predict(_message):
        raise error

    monkeypatch.setattr(intent_detector, "_ML_AVAILABLE", True)
    monkeypatch.setattr(intent_detector, "_ml_predict", predict)
    with pytest.raises(error_type) as caught:
        intent_detector.predict_rag_intent(QUERY)
    assert caught.value is error
