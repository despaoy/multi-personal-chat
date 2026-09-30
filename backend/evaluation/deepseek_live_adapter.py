"""Isolated evaluation adapter; credentials never enter traces or configuration files."""
import asyncio
import time

import httpx

DEFAULT_CLOUD_CONTEXT_TOKENS = 65536


def replay_answer_budget(*, cloud: bool, requested: int | None = None) -> int:
    """Output allowance independent of retrieval size or fixture wording."""
    value = requested if requested is not None else (1024 if cloud else 256)
    if type(value) is not int or not 32 <= value <= 8192:
        raise ValueError('Answer output budget must be between 32 and 8192 tokens')
    return value


def cloud_context_budgets(window=DEFAULT_CLOUD_CONTEXT_TOKENS):
    """Application working budget, not the provider's theoretical 1M maximum.

    Explicit injection keeps local serving limits and concurrent callers intact.
    Final generation and writer serialization still enforce the token window.
    """
    if not 8192 <= window <= 1000000:
        raise ValueError('Cloud context window must be between 8192 and 1000000')
    return dict(history_limit=max(24, min(256, window // 512)),
                history_max_chars=window, source_max_chars=window // 4)


class DeepSeekEvaluationClient:
    def __init__(self, api_key, model):
        self._key = api_key
        self.model = model
        self.calls = []
        self._client = httpx.AsyncClient(timeout=150)

    async def complete(self, messages, *, purpose, max_tokens=256, temperature=.2, top_p=.9,
                       frequency_penalty=0):
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, frequency_penalty=frequency_penalty, stream=False, thinking={'type': 'disabled'})
        start = time.monotonic()
        call = dict(purpose=purpose, request=body, status='started')
        self.calls.append(call)
        try:
            response = await self._client.post('https://api.deepseek.com/chat/completions', json=body,
                headers={'Authorization': 'Bearer '+self._key})
            call['http_status'] = response.status_code
            if response.status_code != 200:
                raise RuntimeError(f'DeepSeek HTTP {response.status_code}')
            result = response.json()
            call['response'] = result
            choice = result['choices'][0]
            if choice['finish_reason'] != 'stop':
                raise RuntimeError('DeepSeek response incomplete: '+choice['finish_reason'])
            call['status'] = 'completed'
            return choice['message']['content']
        except asyncio.CancelledError:
            call['status'] = 'cancelled_or_timed_out'
            raise
        except Exception as exc:
            call.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            call['seconds'] = time.monotonic() - start

    async def generate(self, **kwargs):
        if kwargs.get('lora_name'):
            raise ValueError('No LoRA allowed in cloud evaluation')
        return await self.complete(kwargs['messages'], purpose='answer',
            **{k: kwargs[k] for k in ('max_tokens', 'temperature', 'top_p', 'frequency_penalty')})

    async def close(self):
        await self._client.aclose()


class DeepSeekRecordedWriter:
    def __init__(self, client):
        self.client = client
        self.calls = []

    async def complete(self, messages):
        start = time.monotonic()
        output = await self.client.complete(messages, purpose='writer', max_tokens=768, temperature=0, top_p=1)
        self.calls.append(dict(messages=messages, output=output, seconds=time.monotonic()-start))
        return output

    async def close(self):
        pass  # Shared transport is closed by the evaluation driver.


def cloud_context_components(client, *, context_window_tokens=DEFAULT_CLOUD_CONTEXT_TOKENS):
    """Inject every optional dialogue reviewer explicitly, with purpose traces.

    Keep native trigger/timeout/parsing policies: enabled is not evidence that
    a reviewer ran or that its result was accepted. Never fall back to vLLM.
    """
    from character.contextual_policy import ContextualDecisionPolicy
    from character.evidence_selector import ContextualEvidenceSelector
    from character.semantic_review_adapter import (
        DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS,
        SEMANTIC_REVIEW_MAX_TOKENS,
    )
    from character.semantic_state_estimator import SemanticStateEstimator
    from inference.context_budget import ReviewContextBudget

    budget = ReviewContextBudget(context_window_tokens)

    def reviewer(purpose, max_tokens):
        async def review(messages):
            return await client.complete([dict(item) for item in messages], purpose=purpose,
                max_tokens=max_tokens, temperature=0, top_p=1)
        return review

    return dict(
        semantic_estimator=SemanticStateEstimator(
            reviewer('semantic_review', SEMANTIC_REVIEW_MAX_TOKENS),
            timeout_seconds=DEFAULT_SEMANTIC_REVIEW_TIMEOUT_SECONDS, context_budget=budget),
        memory_selector=ContextualEvidenceSelector(reviewer('memory_selection', 2048), context_budget=budget),
        contextual_policy=ContextualDecisionPolicy(reviewer('contextual_policy', 160), context_budget=budget),
    )
