"""Configuration snapshots do not depend on the unrelated Redis cache."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_config_snapshot_import_and_read_need_no_redis():
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class NoRedis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'redis' or fullname.startswith('redis.') or fullname == 'cache.redis_client':
            raise ModuleNotFoundError('Redis is unrelated to a process-local configuration snapshot')
sys.meta_path.insert(0, NoRedis())
from cache.config_cache import invalidate_config_cache, read_config
class Store:
    reads = 0
    @property
    def config(self):
        self.reads += 1
        return {'useKnowledgeBase': False, 'maxTokens': 256, 'temperature': .7}
store = Store()
first, hit = read_config(store)
assert hit is False and first['maxTokens'] == 256
assert read_config(store) == (first, True) and store.reads == 1
invalidate_config_cache()
assert read_config(store) == (first, False) and store.reads == 2
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(Path(__file__).resolve().parents[1])],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def snapshot(monkeypatch):
    from cache import config_cache

    clock = [0.]
    config_cache.invalidate_config_cache()
    monkeypatch.setattr(config_cache, "monotonic", lambda: clock[0])
    monkeypatch.setattr(config_cache, "_ttl_with_jitter", lambda ttl: ttl)
    yield config_cache, clock
    config_cache.invalidate_config_cache()


class Store:
    def __init__(self, value):
        self.value = value
        self.failure = None
        self.reads = 0

    @property
    def config(self):
        self.reads += 1
        if self.failure is not None:
            raise self.failure
        return self.value


def test_snapshot_hit_copy_invalidation_and_database_identity(snapshot):
    cache, _clock = snapshot
    first = Store({'useKnowledgeBase': True})
    value, hit = cache.read_config(first)
    assert value == first.value and hit is False
    value['useKnowledgeBase'] = False
    assert cache.read_config(first) == (first.value, True) and first.reads == 1
    second = Store({'useKnowledgeBase': False})
    assert cache.read_config(second) == (second.value, False) and second.reads == 1
    cache.invalidate_config_cache()
    assert cache.read_config(second) == (second.value, False) and second.reads == 2


def test_expired_snapshot_failure_never_reuses_old_configuration(snapshot):
    cache, clock = snapshot
    store = Store({'useKnowledgeBase': True})
    assert cache.read_config(store) == (store.value, False)
    clock[0] = 61.
    failure = OSError('Controlled configuration database unavailable')
    store.failure = failure
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            cache.read_config(store)
        assert caught.value is failure
    assert store.reads == 3
    store.failure = None
    store.value = {'useKnowledgeBase': False}
    assert cache.read_config(store) == (store.value, False) and store.reads == 4


def test_bad_configuration_shape_is_not_a_successful_empty_snapshot(snapshot):
    cache, _clock = snapshot
    store = Store([])
    with pytest.raises(TypeError, match='mapping'):
        cache.read_config(store)
    store.value = {'useKnowledgeBase': False}
    assert cache.read_config(store) == (store.value, False) and store.reads == 2
