from __future__ import annotations

import threading
import time

from agent.runtime.parallel import current_active, current_quota, parallel_map_ordered, reset_quota, track_active


def test_parallel_map_ordered_preserves_order_and_runs_concurrently():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker(item: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return item * 2

    result = parallel_map_ordered([1, 2, 3, 4], worker, max_workers=4)

    assert result == [2, 4, 6, 8]
    assert max_active > 1


def test_track_active_updates_counter():
    assert current_active("demo") == 0
    with track_active("demo"):
        assert current_active("demo") == 1
    assert current_active("demo") == 0


def test_reset_quota_clears_global_counter():
    reset_quota("qwen")
    assert current_quota("qwen") == 0
