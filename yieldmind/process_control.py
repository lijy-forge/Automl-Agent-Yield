"""Cancellation-aware process execution with descendant cleanup on POSIX."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable


ProcessTarget = Callable[..., Any]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class ManagedProcessOutcome:
    status: str
    value: Any = None
    error: str = ""
    pid: int | None = None
    exitcode: int | None = None
    duration_seconds: float = 0.0
    terminate_sent: bool = False
    kill_sent: bool = False
    process_group: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("value", None)
        payload["ok"] = self.ok
        return payload


def _managed_worker(result_queue: Any, target: ProcessTarget, args: tuple[Any, ...]) -> None:
    process_group = False
    try:
        if os.name == "posix":
            os.setsid()
            process_group = True
        value = target(*args)
        result_queue.put(
            {
                "ok": True,
                "value": value,
                "process_group": process_group,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20),
                "process_group": process_group,
            }
        )


def _stop_process_tree(
    proc: mp.Process,
    *,
    terminate_grace_seconds: float,
) -> tuple[bool, bool]:
    terminate_sent = False
    kill_sent = False
    if not proc.is_alive():
        proc.join(timeout=0)
        return terminate_sent, kill_sent

    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        terminate_sent = True
    except (ProcessLookupError, OSError):
        if proc.is_alive():
            proc.terminate()
            terminate_sent = True

    proc.join(max(0.0, terminate_grace_seconds))
    if proc.is_alive():
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            elif hasattr(proc, "kill"):
                proc.kill()
            else:
                proc.terminate()
            kill_sent = True
        except (ProcessLookupError, OSError):
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                kill_sent = True
        proc.join(2.0)
    return terminate_sent, kill_sent


def run_managed_process(
    target: ProcessTarget,
    args: tuple[Any, ...] = (),
    *,
    timeout_seconds: float,
    cancel_check: CancellationCheck | None = None,
    poll_interval_seconds: float = 0.1,
    terminate_grace_seconds: float = 3.0,
    start_method: str = "spawn",
) -> ManagedProcessOutcome:
    """Run a picklable target and stop its whole process group when interrupted.

    The target runs in a new POSIX session before user code starts. Descendant
    subprocesses therefore inherit the same process group unless they explicitly
    detach themselves.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero")

    started = time.monotonic()
    ctx = mp.get_context(start_method)
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_managed_worker, args=(result_queue, target, tuple(args)))
    proc.start()
    stop_status = ""
    controller_error = ""

    while proc.is_alive():
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            stop_status = "timed_out"
            break
        if cancel_check is not None:
            try:
                if cancel_check():
                    stop_status = "cancelled"
                    break
            except Exception as exc:
                stop_status = "controller_error"
                controller_error = f"Cancellation check failed: {type(exc).__name__}: {exc}"
                break
        proc.join(min(poll_interval_seconds, max(0.0, timeout_seconds - elapsed)))

    terminate_sent = False
    kill_sent = False
    if stop_status:
        terminate_sent, kill_sent = _stop_process_tree(
            proc,
            terminate_grace_seconds=terminate_grace_seconds,
        )
        result_queue.close()
        return ManagedProcessOutcome(
            status=stop_status,
            error=controller_error,
            pid=proc.pid,
            exitcode=proc.exitcode,
            duration_seconds=round(time.monotonic() - started, 4),
            terminate_sent=terminate_sent,
            kill_sent=kill_sent,
            process_group=os.name == "posix",
        )

    proc.join(timeout=0)
    try:
        payload = result_queue.get(timeout=1.0)
    except queue.Empty:
        return ManagedProcessOutcome(
            status="no_result" if proc.exitcode == 0 else "failed",
            error=f"Managed process exited without a result (exitcode={proc.exitcode}).",
            pid=proc.pid,
            exitcode=proc.exitcode,
            duration_seconds=round(time.monotonic() - started, 4),
            process_group=os.name == "posix",
        )
    finally:
        result_queue.close()

    process_group = bool(payload.get("process_group")) if isinstance(payload, dict) else False
    if not isinstance(payload, dict) or not payload.get("ok"):
        error = payload.get("error", "Managed process returned an invalid payload.") if isinstance(payload, dict) else str(payload)
        return ManagedProcessOutcome(
            status="failed",
            error=str(error),
            pid=proc.pid,
            exitcode=proc.exitcode,
            duration_seconds=round(time.monotonic() - started, 4),
            process_group=process_group,
        )
    return ManagedProcessOutcome(
        status="completed",
        value=payload.get("value"),
        pid=proc.pid,
        exitcode=proc.exitcode,
        duration_seconds=round(time.monotonic() - started, 4),
        process_group=process_group,
    )
