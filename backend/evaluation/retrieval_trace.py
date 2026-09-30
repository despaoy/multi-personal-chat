"""Scoped retrieval observation for isolated replay; never scores answers."""
import time


class RecordedRetrieval:
    def __init__(self, retrieve, current, *, top_k=None):
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 32):
            raise ValueError('Experimental RAG top_k must be between 1 and 32')
        self.retrieve = retrieve
        self.current = current
        self.top_k = top_k

    async def __call__(self, query, top_k, filters):
        effective = top_k if self.top_k is None else self.top_k
        record = dict(query=query, requested_top_k=top_k, effective_top_k=effective,
                      status='started')
        self.current.setdefault('retrieval_calls', []).append(record)
        started = time.monotonic()
        try:
            bundle = await self.retrieve(query, effective, filters)
            record.update(status='completed', bundle=bundle)
            return bundle
        except BaseException as exc:
            record.update(status='failed_or_cancelled', error_type=type(exc).__name__)
            raise
        finally:
            record['seconds'] = time.monotonic() - started
