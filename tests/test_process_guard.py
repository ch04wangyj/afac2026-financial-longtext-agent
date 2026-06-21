"""重型 Python 进程守卫测试。"""

import pytest

from agent.runtime.process_guard import assert_no_other_heavy_python_jobs, find_heavy_python_jobs


class _FakeMem:
    def __init__(self, rss):
        self.rss = rss


class _FakeProc:
    def __init__(self, pid, name, rss, cmdline):
        self.info = {
            "pid": pid,
            "name": name,
            "memory_info": _FakeMem(rss),
            "cmdline": cmdline,
        }


def test_find_heavy_python_jobs_filters_current_process_and_threshold(monkeypatch):
    procs = [
        _FakeProc(101, "python.exe", int(1.6 * 1024**3), ["python", "worker_a.py"]),
        _FakeProc(202, "python.exe", int(0.5 * 1024**3), ["python", "worker_b.py"]),
        _FakeProc(303, "node.exe", int(2.0 * 1024**3), ["node", "server.js"]),
        _FakeProc(404, "python.exe", int(2.0 * 1024**3), ["python", "self.py"]),
    ]
    monkeypatch.setattr("agent.runtime.process_guard.psutil.process_iter", lambda *args, **kwargs: procs)

    rows = find_heavy_python_jobs(current_pid=404, min_rss_gb=1.0)

    assert rows == [{"pid": 101, "rss_gb": 1.6, "cmdline": "python worker_a.py"}]


def test_assert_no_other_heavy_python_jobs_raises_on_conflict(monkeypatch):
    monkeypatch.setattr(
        "agent.runtime.process_guard.find_heavy_python_jobs",
        lambda current_pid=None, min_rss_gb=1.0: [{"pid": 999, "rss_gb": 1.8, "cmdline": "python worker.py"}],
    )

    with pytest.raises(RuntimeError, match="Heavy Python job already running"):
        assert_no_other_heavy_python_jobs()
