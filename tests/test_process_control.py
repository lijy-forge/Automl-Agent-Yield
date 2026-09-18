from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from yieldmind.process_control import run_managed_process


def _return_value(value: str) -> dict[str, str]:
    return {"value": value}


def _spawn_sleeping_descendant(pid_path: str, sleep_seconds: float) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({sleep_seconds!r})"],
    )
    Path(pid_path).write_text(str(child.pid), encoding="utf-8")
    child.wait()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_file(path: Path, timeout_seconds: float = 3.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return int(path.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise AssertionError(f"Descendant PID file was not created: {path}")


def _wait_until_gone(pid: int, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.05)
    return not _pid_exists(pid)


def test_managed_process_returns_serializable_value() -> None:
    outcome = run_managed_process(_return_value, ("ok",), timeout_seconds=3)

    assert outcome.ok
    assert outcome.status == "completed"
    assert outcome.value == {"value": "ok"}
    assert outcome.exitcode == 0


@pytest.mark.skipif(os.name != "posix", reason="process-group descendant cleanup requires POSIX")
def test_managed_process_timeout_stops_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "timeout-descendant.pid"

    outcome = run_managed_process(
        _spawn_sleeping_descendant,
        (str(pid_path), 30.0),
        timeout_seconds=0.6,
        poll_interval_seconds=0.03,
        terminate_grace_seconds=0.2,
    )
    descendant_pid = _wait_for_pid_file(pid_path)

    assert outcome.status == "timed_out"
    assert outcome.terminate_sent is True
    assert outcome.process_group is True
    assert _wait_until_gone(descendant_pid), f"descendant {descendant_pid} survived timeout"


@pytest.mark.skipif(os.name != "posix", reason="process-group descendant cleanup requires POSIX")
def test_managed_process_cancellation_stops_descendant(tmp_path: Path) -> None:
    pid_path = tmp_path / "cancel-descendant.pid"
    started = time.monotonic()

    def cancel_when_descendant_started() -> bool:
        return pid_path.exists()

    outcome = run_managed_process(
        _spawn_sleeping_descendant,
        (str(pid_path), 30.0),
        timeout_seconds=10,
        cancel_check=cancel_when_descendant_started,
        poll_interval_seconds=0.03,
        terminate_grace_seconds=0.2,
    )
    descendant_pid = _wait_for_pid_file(pid_path)

    assert outcome.status == "cancelled"
    assert outcome.duration_seconds < 3.0
    assert time.monotonic() - started < 3.5
    assert outcome.terminate_sent is True
    assert _wait_until_gone(descendant_pid), f"descendant {descendant_pid} survived cancellation"


def test_legacy_operation_entry_maps_managed_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    import run_yield
    from yieldmind.process_control import ManagedProcessOutcome

    monkeypatch.setattr(
        run_yield,
        "run_managed_process",
        lambda *args, **kwargs: ManagedProcessOutcome(
            status="cancelled",
            pid=123,
            exitcode=-15,
            terminate_sent=True,
            process_group=True,
        ),
    )

    result = run_yield._run_operation_agent_with_process_timeout(
        {},
        "openai",
        "/fake",
        "yield_stress_regression",
        "fake instructions",
        1,
        cancel_check=lambda: True,
    )

    assert result["rcode"] == 130
    assert result["cancelled"] is True
    assert result["timed_out"] is False
    assert result["process_control"]["terminate_sent"] is True


def test_generated_script_execution_uses_argv_for_path_with_spaces(tmp_path: Path) -> None:
    from operation_agent.execution import execute_script

    script_dir = tmp_path / "path with spaces"
    script_dir.mkdir()
    script = script_dir / "generated script.py"
    script.write_text("print('argv execution ok')\n", encoding="utf-8")

    returncode, output = execute_script(str(script.resolve()), work_dir=".")

    assert returncode == 0
    assert "argv execution ok" in output
