"""共享线程池、并发限流与顺序保持的并发辅助。"""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")


_SEMAPHORES: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_SEMAPHORES_LOCK = threading.Lock()
_ACTIVE_COUNTS: dict[str, int] = defaultdict(int)
_ACTIVE_LOCK = threading.Lock()
_QUOTA_COUNTS: dict[str, int] = defaultdict(int)
_QUOTA_LOCK = threading.Lock()


def parallel_map_ordered(items: Iterable[T], fn: Callable[[T], R], max_workers: int) -> list[R]:
    """并发执行 fn，但按输入顺序返回结果。"""
    indexed_items = list(enumerate(items))
    if not indexed_items:
        return []
    results: list[R | None] = [None] * len(indexed_items)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {executor.submit(fn, item): idx for idx, item in indexed_items}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return [item for item in results if item is not None]


@contextmanager
def acquire_named_permit(name: str, limit: int):
    """获取可重入共享信号量，限制全局并发请求数。"""
    sem = _get_named_semaphore(name, limit)
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


@contextmanager
def acquire_named_quota(name: str, limit: int):
    """消耗全局请求配额；超过上限时抛错。"""
    key = name
    with _QUOTA_LOCK:
        current = _QUOTA_COUNTS[key]
        if current >= max(1, int(limit)):
            raise RuntimeError(f"quota exceeded for {name}: {current}/{limit}")
        _QUOTA_COUNTS[key] = current + 1
    try:
        yield
    finally:
        pass


@contextmanager
def track_active(name: str):
    """测试与观测用活动计数器。"""
    with _ACTIVE_LOCK:
        _ACTIVE_COUNTS[name] += 1
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_COUNTS[name] -= 1


def current_active(name: str) -> int:
    with _ACTIVE_LOCK:
        return _ACTIVE_COUNTS.get(name, 0)


def current_quota(name: str) -> int:
    with _QUOTA_LOCK:
        return _QUOTA_COUNTS.get(name, 0)


def reset_quota(name: str | None = None) -> None:
    with _QUOTA_LOCK:
        if name is None:
            _QUOTA_COUNTS.clear()
        else:
            _QUOTA_COUNTS.pop(name, None)


def _get_named_semaphore(name: str, limit: int) -> threading.BoundedSemaphore:
    key = (name, max(1, int(limit)))
    with _SEMAPHORES_LOCK:
        sem = _SEMAPHORES.get(key)
        if sem is None:
            sem = threading.BoundedSemaphore(key[1])
            _SEMAPHORES[key] = sem
        return sem
