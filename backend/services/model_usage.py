"""Account independent model operations before publishing their completion."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from inference.openai_protocol import ProviderResponseError
from inference.usage import collect_usage, has_pending_usage, invocation_usage
from infra.observability import increment

logger = logging.getLogger(__name__)
_WRITE_TIMEOUT = float(os.getenv("DB_WRITE_TIMEOUT", "3"))


class UsageRecordingError(RuntimeError):
    http_status = 503

    def __init__(self, recording_error, operation_error):
        super().__init__("Model invocation recording failed or was not confirmed")
        self.recording_error = recording_error
        self.operation_error = operation_error

    def public_detail(self):
        detail = {"kind": "usage_record_failed", "stage": "model_invocation_record",
                  "message": "模型调用记录保存失败或未确认，请检查数据库后重试",
                  "recordErrorType": type(self.recording_error).__name__, "invocationRecorded": None,
                  "generationErrorType": type(self.operation_error).__name__ if self.operation_error else None}
        if isinstance(self.operation_error, ProviderResponseError):
            detail["providerError"] = self.operation_error.public_detail()
        return detail


async def write_invocation(database, metadata):
    record = {**metadata, **invocation_usage(record_started=True)}
    await asyncio.to_thread(database.add_model_invocation, record)


@asynccontextmanager
async def account_usage(database, metadata):
    with collect_usage():
        started = time.perf_counter()
        error = None

        async def confirm():
            if not has_pending_usage():
                return
            record = {**metadata, "costTime": round(time.perf_counter() - started, 2),
                      "errorType": type(error.__cause__ or error).__name__ if error else ""}
            try:
                await asyncio.wait_for(write_invocation(database, record), timeout=_WRITE_TIMEOUT)
            except Exception as exc:
                increment("db_write_failures")
                logger.warning("Invocation record failed purpose=%s traceId=%s type=%s",
                               metadata["purpose"], metadata["traceId"], type(exc).__name__)
                failure = UsageRecordingError(exc, error)
                if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
                    raise error from failure
                raise failure from exc

        try:
            yield confirm
        except (Exception, asyncio.CancelledError, GeneratorExit) as exc:
            error = exc
            raise
        finally:
            await confirm()
