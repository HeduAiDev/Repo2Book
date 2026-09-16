# ch34 测试 conftest：gloo 进程池的饥饿关闭——模块级 fixture 默认活到 session
# 结束，8+4+2+1 个 spawn worker（每个 ~500MB、torch import 慢）同时驻留会把
# 后续测试拖挂（Windows 下进程/内存压力 → gloo rendezvous 超时或挂死）。
# 这里按『最后一个消费该池的测试』就地下闸：每池用完立即 close，进程数峰值
# 从 15 降到单池规模。
from __future__ import annotations

import pytest

from _pool import GlooPool

_POOLS = {"pool8": 8, "pool2": 2, "pool4": 4, "pool1": 1}
_live: dict[str, GlooPool] = {}
_last_use: dict[str, int] = {}


def pytest_collection_modifyitems(items):
    _last_use.clear()
    for i, item in enumerate(items):
        for name in _POOLS:
            if name in item.fixturenames:
                _last_use[name] = i


@pytest.fixture(scope="module")
def pool8():
    _live["pool8"] = GlooPool(8)
    yield _live["pool8"]


@pytest.fixture(scope="module")
def pool2():
    _live["pool2"] = GlooPool(2)
    yield _live["pool2"]


@pytest.fixture(scope="module")
def pool4():
    _live["pool4"] = GlooPool(4)
    yield _live["pool4"]


@pytest.fixture(scope="module")
def pool1():
    _live["pool1"] = GlooPool(1)
    yield _live["pool1"]


def pytest_runtest_teardown(item, nextitem):
    idx = getattr(item, "item_index", None)
    if idx is None:
        return
    for name in list(_live):
        if _last_use.get(name) == idx:
            pool = _live.pop(name)
            try:
                pool.close()
            except Exception:
                for p in pool.procs:
                    if p.is_alive():
                        p.kill()


def pytest_collection_finish(session):
    for i, item in enumerate(session.items):
        item.item_index = i
