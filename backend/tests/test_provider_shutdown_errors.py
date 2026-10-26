"""Model shutdown must expose close failures after attempting every provider."""

import asyncio
import logging
from unittest.mock import Mock

import pytest

from app import main
from inference import model_manager
from test_owned_startup_resources import lifecycle as owned_lifecycle

lifecycle = owned_lifecycle


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.setenv('MODEL_PROVIDER', 'openai_compat')
    monkeypatch.setattr(model_manager, '_get_db_config', lambda _owner_database=None: {'modelProvider': 'openai_compat', 'openaiCompatBaseUrl': 'https://synthetic.invalid', 'openaiCompatApiKey': 'synthetic-key', 'openaiCompatModel': 'synthetic-model'})
    return model_manager.ModelManager(base_dir=tmp_path)


def exception_chain(error):
    chain = []
    while error is not None:
        chain.append(error)
        error = error.__context__
    return chain


def controlled_closes(manager, failures, monkeypatch):
    order = []
    providers = list(manager._providers.values())
    for index, provider in enumerate(providers):

        def close(index=index):
            order.append(index)
            if index in failures:
                raise failures[index]

        monkeypatch.setattr(provider, "close", Mock(side_effect=close))
    return order, providers


@pytest.mark.parametrize("failed", [(), (0,), (3,), (5,), (0, 3, 5)])
def test_every_provider_is_attempted_in_order_and_failures_propagate(manager, monkeypatch, caplog, failed):
    failures = {index: RuntimeError("synthetic-private-close-detail-" + str(index)) for index in failed}
    order, providers = controlled_closes(manager, failures, monkeypatch)
    caplog.set_level(logging.WARNING, logger="inference.model_manager")
    if failed:
        with pytest.raises(model_manager.ProviderShutdownError) as raised:
            manager.shutdown()
        assert raised.value.failures == tuple((providers[index].name, failures[index]) for index in failed)
        assert all(error in [entry[1] for entry in raised.value.failures] for error in failures.values())
        assert "synthetic-private" not in str(raised.value)
    else:
        assert manager.shutdown() is None
    assert order == list(range(len(providers)))
    for provider in providers:
        provider.close.assert_called_once()
    assert "synthetic-private" not in caplog.text


@pytest.mark.parametrize("close_fails", [False, True])
def test_shutdown_preserves_callers_active_failure_and_success_does_not_reraise_it(manager, monkeypatch, close_fails):
    primary = ValueError("synthetic primary failure")
    close_error = RuntimeError("synthetic provider close failure")
    order, providers = controlled_closes(manager, {0: close_error} if close_fails else {}, monkeypatch)
    try:
        raise primary
    except ValueError:
        if close_fails:
            with pytest.raises(model_manager.ProviderShutdownError) as raised:
                manager.shutdown()
            assert raised.value.failures == ((providers[0].name, close_error),)
            assert primary in exception_chain(raised.value)
        else:
            assert manager.shutdown() is None
    assert order == list(range(len(providers)))


def test_provider_cancellation_is_propagated_after_remaining_closes(manager, monkeypatch):
    failure = asyncio.CancelledError()
    order, providers = controlled_closes(manager, {1: failure}, monkeypatch)
    with pytest.raises(asyncio.CancelledError) as raised:
        manager.shutdown()
    assert raised.value is failure
    assert isinstance(raised.value.__cause__, model_manager.ProviderShutdownError)
    assert raised.value.__cause__.failures == ((providers[1].name, failure),)
    assert order == list(range(len(providers)))


def test_unloaded_real_providers_close_without_creating_model_or_http_resources(manager):
    assert len(manager._providers) == 6
    local = manager._providers[model_manager.ModelProvider.TRANSFORMERS_PEFT]
    assert local._model is local._tokenizer is None
    manager.shutdown()
    assert local._model is local._tokenizer is None
    assert local._loaded is False


@pytest.mark.parametrize("startup_failure", [False, True])
async def test_real_manager_close_failure_reaches_lifespan_after_other_cleanup(
    manager, lifecycle, monkeypatch, caplog, startup_failure
):
    first = RuntimeError("synthetic-private-provider-failure")
    second = ValueError("synthetic-private-last-provider-failure")
    order, providers = controlled_closes(manager, {0: first, 5: second}, monkeypatch)
    if startup_failure:
        lifecycle.worker.side_effect = lifecycle.failure
    app, database = lifecycle.application()
    database._model_manager = manager
    with pytest.raises(model_manager.ProviderShutdownError) as raised:
        async with main.lifespan(app):
            if startup_failure:
                pytest.fail("Startup failure admitted requests")
    assert raised.value.failures == ((providers[0].name, first), (providers[5].name, second))
    chain = exception_chain(raised.value)
    if startup_failure:
        assert lifecycle.failure in chain
    assert order == list(range(len(providers)))
    assert lifecycle.events == ["http", "backup"]
    assert all(getattr(lifecycle.config, name) is None for name in main._LIFESPAN_RESOURCE_NAMES)
    assert "synthetic-private" not in caplog.text
    assert "资源清理完成" not in caplog.text


def test_cancellation_with_additional_close_failure_retains_both(manager, monkeypatch):
    cancellation = asyncio.CancelledError()
    close_error = RuntimeError("synthetic-private-additional-close-failure")
    order, providers = controlled_closes(manager, {1: cancellation, 4: close_error}, monkeypatch)
    with pytest.raises(asyncio.CancelledError) as raised:
        manager.shutdown()
    assert raised.value is cancellation
    aggregate = raised.value.__cause__
    assert isinstance(aggregate, model_manager.ProviderShutdownError)
    assert aggregate.failures == ((providers[1].name, cancellation), (providers[4].name, close_error))
    assert "synthetic-private" not in str(aggregate)
    assert order == list(range(len(providers)))
