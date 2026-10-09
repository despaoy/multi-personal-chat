"""Memory vector cache is bounded and only publishes successful batches."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from character.memory_service import CharacterMemoryService


def record(i):
    return dict(id=i,updated_at='2026-10-09T00:00:00+00:00',content=f'complete-memory-{i}')


def test_memory_vector_lru_keeps_recent_hits_and_reencodes_evicted_records():
    encode=Mock(side_effect=lambda texts:np.ones((len(texts),2),dtype=np.float32))
    service=CharacterMemoryService(Mock(),embedding_provider=SimpleNamespace(embed_texts=encode))
    records=[record(i) for i in range(1024)]
    assert len(service._semantic_similarities('query',records))==1024
    service._semantic_similarities('query',[record(0)])
    service._semantic_similarities('query',[record(1024)])
    service._semantic_similarities('query',[record(0)])
    assert encode.call_args.args[0]==['query']
    service._semantic_similarities('query',[record(1)])
    assert encode.call_args.args[0]==['query','complete-memory-1']


def test_failed_batch_does_not_publish_partial_vectors():
    calls=[]
    fail=False
    def encode(texts):
        calls.append(list(texts))
        matrix=np.ones((len(texts),2),dtype=np.float32)
        if fail:
            matrix[-1,0]=np.nan
        return matrix
    service=CharacterMemoryService(Mock(),embedding_provider=SimpleNamespace(embed_texts=encode))
    service._semantic_similarities('query',[record(0)])
    fail=True
    with pytest.raises(ValueError,match='nonfinite'):
        service._semantic_similarities('query',[record(1),record(2)])
    fail=False
    result=service._semantic_similarities('query',[record(1),record(2)])
    assert calls[-1]==['query','complete-memory-1','complete-memory-2']
    assert len(result)==2 and all(score==pytest.approx(1) for score in result.values())
    service._semantic_similarities('query',[record(0)])
    assert calls[-1]==['query']
