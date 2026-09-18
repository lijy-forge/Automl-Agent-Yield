"""YieldMind persistence with PostgreSQL production and SQLite offline modes."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "yieldmind.sqlite3"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
_ENGINES: dict[str, Engine] = {}

TASK_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"queued"}),
    "queued": frozenset({"running", "dispatch_failed", "cancelled"}),
    "running": frozenset({"completed", "failed", "cancel_requested", "interrupted"}),
    "cancel_requested": frozenset({"cancelled", "completed", "failed", "interrupted"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "dispatch_failed": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            number = float(value)
            return number if number == number and abs(number) != float("inf") else None
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload if payload is not None else {}, ensure_ascii=False, default=_json_default)


def json_loads(raw: str | bytes | None) -> Any:
    if raw in (None, "", b""):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def default_db_path() -> Path:
    raw = os.environ.get("YIELDMIND_DB_PATH")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DB_PATH


def default_database_target() -> str | Path:
    return os.environ.get("YIELDMIND_DATABASE_URL") or default_db_path()


def database_backend(target: str | Path | None = None) -> str:
    value = target if target is not None else default_database_target()
    if isinstance(value, Path):
        return "sqlite"
    raw = str(value)
    if raw.startswith("postgresql"):
        return "postgresql"
    if raw.startswith("sqlite:"):
        return "sqlite"
    return "sqlite"


def database_location(target: str | Path) -> str:
    if database_backend(target) == "postgresql":
        return make_url(str(target)).render_as_string(hide_password=True)
    return str(target)


class _SQLAlchemyResult:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.rowcount = int(result.rowcount or 0)
        self.lastrowid = None

    def fetchone(self) -> Any:
        return self._result.mappings().first()

    def fetchall(self) -> list[Any]:
        return list(self._result.mappings().all())


def _named_statement(sql: str, params: Iterable[Any]) -> tuple[str, dict[str, Any]]:
    values = list(params)
    pieces = sql.split("?")
    if len(pieces) - 1 != len(values):
        raise ValueError(f"SQL placeholder count does not match parameter count: {sql[:120]}")
    rendered = pieces[0]
    bound: dict[str, Any] = {}
    for index, value in enumerate(values):
        key = f"p{index}"
        rendered += f":{key}{pieces[index + 1]}"
        bound[key] = value
    return rendered, bound


def _engine(database_url: str) -> Engine:
    engine = _ENGINES.get(database_url)
    if engine is None:
        engine = create_engine(database_url, pool_pre_ping=True, future=True)
        _ENGINES[database_url] = engine
    return engine


class _SQLAlchemyConnection:
    def __init__(self, database_url: str) -> None:
        self.engine = _engine(database_url)
        self.connection: Any = None
        self.transaction: Any = None

    def __enter__(self) -> "_SQLAlchemyConnection":
        self.connection = self.engine.connect()
        self.transaction = self.connection.begin()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                self.transaction.commit()
            else:
                self.transaction.rollback()
        finally:
            self.connection.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> _SQLAlchemyResult:
        statement, bound = _named_statement(sql, params)
        return _SQLAlchemyResult(self.connection.execute(text(statement), bound))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> _SQLAlchemyResult:
        row_list = [list(row) for row in rows]
        if not row_list:
            return _SQLAlchemyResult(self.connection.execute(text("SELECT 1 WHERE 1=0")))
        statement, _ = _named_statement(sql, row_list[0])
        bound_rows = []
        for row in row_list:
            _, bound = _named_statement(sql, row)
            bound_rows.append(bound)
        return _SQLAlchemyResult(self.connection.execute(text(statement), bound_rows))


def connect(db_path: str | Path | None = None) -> Any:
    target = db_path if db_path is not None else default_database_target()
    if database_backend(target) == "postgresql":
        return _SQLAlchemyConnection(str(target))
    raw_path = str(target)
    if raw_path.startswith("sqlite:///"):
        raw_path = raw_path.removeprefix("sqlite:///")
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_db(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
            conn.executescript(migration.read_text(encoding="utf-8"))
        _ensure_optional_columns(conn)
        _ensure_task_optional_columns(conn)
    return path


def initialize_postgres(database_url: str) -> str:
    from alembic import command
    from alembic.config import Config

    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")
    return database_url


def initialize_database(target: str | Path | None = None) -> str | Path:
    resolved = target if target is not None else default_database_target()
    if database_backend(resolved) == "postgresql":
        return initialize_postgres(str(resolved))
    return initialize_db(resolved)


def _ensure_optional_columns(conn: sqlite3.Connection) -> None:
    """Apply small additive migrations that SQLite cannot express with IF NOT EXISTS."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(yieldmind_tool_calls)").fetchall()}
    additions = {
        "session_id": "TEXT NOT NULL DEFAULT ''",
        "turn_id": "TEXT NOT NULL DEFAULT ''",
        "idempotency_key": "TEXT NOT NULL DEFAULT ''",
    }
    for name, ddl in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE yieldmind_tool_calls ADD COLUMN {name} {ddl}")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_yieldmind_tool_calls_idempotency
        ON yieldmind_tool_calls(idempotency_key)
        WHERE idempotency_key <> ''
        """
    )


def _ensure_task_optional_columns(conn: sqlite3.Connection) -> None:
    """Add task audit and lease columns to older SQLite databases."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(yieldmind_tasks)").fetchall()}
    additions = {
        "cancel_requested_at": "REAL",
        "cancelled_at": "REAL",
        "cancel_reason": "TEXT NOT NULL DEFAULT ''",
        "cancel_requested_by": "TEXT NOT NULL DEFAULT ''",
        "worker_id": "TEXT NOT NULL DEFAULT ''",
        "heartbeat_at": "REAL",
        "lease_expires_at": "REAL",
        "interrupted_at": "REAL",
        "recovery_reason": "TEXT NOT NULL DEFAULT ''",
        "recovery_requested_by": "TEXT NOT NULL DEFAULT ''",
        "recovery_of_task_id": "TEXT NOT NULL DEFAULT ''",
    }
    for name, ddl in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE yieldmind_tasks ADD COLUMN {name} {ddl}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_yieldmind_tasks_lease ON yieldmind_tasks(status, lease_expires_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_yieldmind_tasks_recovery ON yieldmind_tasks(recovery_of_task_id)"
    )


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for key in ("metadata_json", "result_json", "payload_json", "args_json", "summary_json"):
        if key in payload:
            payload[key.replace("_json", "")] = json_loads(payload.pop(key))
    return payload


class YieldMindStore:
    """Small repository object used by scripts, API handlers, and tests."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        database_url: str | None = None,
        initialize: bool | None = None,
    ) -> None:
        target: str | Path = database_url or (db_path if db_path is not None else default_database_target())
        self.backend = database_backend(target)
        should_initialize = self.backend == "sqlite" if initialize is None else initialize
        self.db_path = initialize_database(target) if should_initialize else target
        self.database_url = str(self.db_path) if self.backend == "postgresql" else f"sqlite:///{self.db_path}"
        self.location = database_location(self.db_path)

    def ping(self) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 AS ok").fetchone()
        return bool(row and int(row["ok"]) == 1)

    def create_run(
        self,
        *,
        mode: str,
        source: str,
        status: str = "created",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_runs
                    (run_id, status, mode, source, started_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, status, mode, source, now, now, json_dumps(metadata or {})),
            )
        return run_id

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        completed: bool = False,
    ) -> None:
        existing = self.get_run(run_id)
        if not existing:
            raise KeyError(f"Unknown run_id: {run_id}")
        now = time.time()
        next_status = status or existing["status"]
        next_result = result if result is not None else existing.get("result", {})
        next_metadata = metadata if metadata is not None else existing.get("metadata", {})
        completed_at = now if completed else existing.get("completed_at")
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE yieldmind_runs
                SET status=?, updated_at=?, completed_at=?, metadata_json=?, result_json=?
                WHERE run_id=?
                """,
                (
                    next_status,
                    now,
                    completed_at,
                    json_dumps(next_metadata),
                    json_dumps(next_result),
                    run_id,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM yieldmind_runs WHERE run_id=?", (run_id,)).fetchone()
        return _row_to_dict(row)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM yieldmind_runs ORDER BY started_at DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def add_event(
        self,
        run_id: str,
        *,
        stage: str,
        message: str,
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> int:
        with connect(self.db_path) as conn:
            sql = """
                INSERT INTO yieldmind_events
                    (run_id, ts, stage, level, message, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """
            if self.backend == "postgresql":
                sql += " RETURNING event_id"
            cur = conn.execute(
                sql,
                (run_id, time.time(), stage, level, message, json_dumps(payload or {})),
            )
            if self.backend == "postgresql":
                row = cur.fetchone()
                return int(row["event_id"])
            return int(cur.lastrowid)

    def list_events(self, run_id: str, *, after_event_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM yieldmind_events
                WHERE run_id=? AND event_id>?
                ORDER BY event_id ASC
                LIMIT ?
                """,
                (run_id, int(after_event_id), max(1, min(int(limit), 1000))),
            ).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def record_stage_execution(
        self,
        *,
        stage_execution_id: str,
        run_id: str,
        thread_id: str,
        stage: str,
        status: str,
        input_hash: str,
        started_at: float,
        completed_at: float | None = None,
        payload: dict[str, Any] | None = None,
        error: str = "",
    ) -> str:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_stage_executions
                    (stage_execution_id, run_id, thread_id, stage, status, input_hash,
                     started_at, completed_at, payload_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stage_execution_id,
                    run_id,
                    thread_id,
                    stage,
                    status,
                    input_hash,
                    started_at,
                    completed_at or time.time(),
                    json_dumps(payload or {}),
                    str(error or ""),
                ),
            )
        return stage_execution_id

    def list_stage_executions(self, run_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM yieldmind_stage_executions
                WHERE run_id=?
                ORDER BY started_at ASC
                LIMIT ?
                """,
                (run_id, max(1, min(int(limit), 2000))),
            ).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def record_tool_call(
        self,
        *,
        tool_name: str,
        mode: str,
        status: str,
        args: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: str = "",
        run_id: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        started_at: float | None = None,
        completed_at: float | None = None,
    ) -> str:
        call_id = f"tool_{uuid.uuid4().hex[:12]}"
        started = started_at or time.time()
        completed = completed_at or time.time()
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_tool_calls
                    (call_id, run_id, tool_name, mode, status, started_at, completed_at,
                     args_json, result_json, error, session_id, turn_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    run_id,
                    tool_name,
                    mode,
                    status,
                    started,
                    completed,
                    json_dumps(args),
                    json_dumps(result or {}),
                    str(error or ""),
                    str(session_id or ""),
                    str(turn_id or ""),
                ),
            )
        return call_id

    def claim_tool_call(
        self,
        *,
        tool_name: str,
        mode: str,
        args: dict[str, Any],
        idempotency_key: str,
        run_id: str | None = None,
        started_at: float | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically reserve an idempotent tool call before side effects run."""
        if not idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        with connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT * FROM yieldmind_tool_calls WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        if existing:
            return _row_to_dict(existing) or {}, True

        call_id = f"tool_{uuid.uuid4().hex[:12]}"
        started = started_at or time.time()
        try:
            with connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO yieldmind_tool_calls
                        (call_id, run_id, tool_name, mode, status, started_at, completed_at,
                         args_json, result_json, error, session_id, turn_id, idempotency_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id,
                        run_id,
                        tool_name,
                        mode,
                        "running",
                        started,
                        None,
                        json_dumps(args),
                        "{}",
                        "",
                        "",
                        "",
                        idempotency_key,
                    ),
                )
        except (IntegrityError, sqlite3.IntegrityError):
            with connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT * FROM yieldmind_tool_calls WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
            if existing:
                return _row_to_dict(existing) or {}, True
            raise
        return self.get_tool_call(call_id) or {}, False

    def get_tool_call(self, call_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM yieldmind_tool_calls WHERE call_id=?", (call_id,)).fetchone()
        return _row_to_dict(row)

    def complete_tool_call(
        self,
        call_id: str,
        *,
        status: str,
        mode: str,
        result: dict[str, Any],
        error: str = "",
        completed_at: float | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tool_calls
                SET status=?, mode=?, result_json=?, error=?, completed_at=?
                WHERE call_id=? AND status='running'
                """,
                (status, mode, json_dumps(result), str(error or ""), completed_at or time.time(), call_id),
            )
        if cur.rowcount == 0:
            raise KeyError(f"Running tool call not found: {call_id}")

    def create_eval_run(
        self,
        *,
        mode: str,
        summary: dict[str, Any],
        report_path: str = "",
        started_at: float | None = None,
        completed_at: float | None = None,
    ) -> str:
        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        started = started_at or time.time()
        completed = completed_at or time.time()
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_eval_runs
                    (eval_id, mode, started_at, completed_at, summary_json, report_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (eval_id, mode, started, completed, json_dumps(summary), report_path),
            )
        return eval_id

    def create_task(
        self,
        *,
        task_type: str,
        idempotency_key: str,
        payload: dict[str, Any],
        recovery_of_task_id: str = "",
    ) -> tuple[dict[str, Any], bool]:
        with connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT * FROM yieldmind_tasks WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        if existing:
            return _row_to_dict(existing) or {}, True
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        now = time.time()
        try:
            with connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO yieldmind_tasks
                        (task_id, task_type, status, idempotency_key, payload_json,
                         result_json, run_id, error, recovery_of_task_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        task_type,
                        "created",
                        idempotency_key,
                        json_dumps(payload),
                        "{}",
                        "",
                        "",
                        str(recovery_of_task_id or ""),
                        now,
                        now,
                    ),
                )
        except (IntegrityError, sqlite3.IntegrityError):
            with connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT * FROM yieldmind_tasks WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
            if existing:
                return _row_to_dict(existing) or {}, True
            raise
        return self.get_task(task_id) or {}, False

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM yieldmind_tasks WHERE task_id=?", (task_id,)).fetchone()
        return _row_to_dict(row)

    def get_task_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM yieldmind_tasks WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        return _row_to_dict(row)

    def update_task(
        self,
        task_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        run_id: str = "",
        error: str = "",
    ) -> None:
        existing = self.get_task(task_id)
        if existing is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        current_status = str(existing["status"])
        if status != current_status and status not in TASK_STATUS_TRANSITIONS.get(current_status, frozenset()):
            raise ValueError(f"Illegal task status transition: {current_status} -> {status}")
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET status=?, result_json=?, run_id=COALESCE(NULLIF(?, ''), run_id),
                    error=?, updated_at=?
                WHERE task_id=? AND status=?
                """,
                (status, json_dumps(result or {}), run_id, error, time.time(), task_id, current_status),
            )
        if cur.rowcount == 0:
            raise RuntimeError(f"Task {task_id} changed concurrently from expected state {current_status}.")

    def transition_task(
        self,
        task_id: str,
        *,
        from_statuses: Iterable[str],
        to_status: str,
        result: dict[str, Any] | None = None,
        run_id: str = "",
        error: str = "",
    ) -> bool:
        """Atomically change task state only when its current state is expected."""
        allowed = tuple(dict.fromkeys(str(status) for status in from_statuses if status))
        if not allowed:
            raise ValueError("from_statuses must contain at least one status")
        illegal = [status for status in allowed if to_status not in TASK_STATUS_TRANSITIONS.get(status, frozenset())]
        if illegal:
            raise ValueError(f"Illegal task status transition to {to_status} from: {', '.join(illegal)}")
        placeholders = ", ".join("?" for _ in allowed)
        clear_lease = to_status in {"completed", "failed", "dispatch_failed", "cancelled", "interrupted"}
        lease_assignment = ", lease_expires_at=NULL" if clear_lease else ""
        with connect(self.db_path) as conn:
            cur = conn.execute(
                f"""
                UPDATE yieldmind_tasks
                SET status=?, result_json=?, run_id=COALESCE(NULLIF(?, ''), run_id),
                    error=?, updated_at=?{lease_assignment}
                WHERE task_id=? AND status IN ({placeholders})
                """,
                (
                    to_status,
                    json_dumps(result or {}),
                    run_id,
                    error,
                    time.time(),
                    task_id,
                    *allowed,
                ),
            )
        return cur.rowcount == 1

    def cancel_queued_task_record(
        self,
        task_id: str,
        *,
        reason: str = "",
        requested_by: str = "",
        requested_at: float | None = None,
    ) -> bool:
        """Atomically cancel a queued task and persist cancellation audit fields."""
        now = requested_at or time.time()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET status='cancelled', cancel_requested_at=?, cancelled_at=?,
                    cancel_reason=?, cancel_requested_by=?, updated_at=?
                WHERE task_id=? AND status='queued'
                """,
                (now, now, str(reason or ""), str(requested_by or ""), now, task_id),
            )
        return cur.rowcount == 1

    def request_running_task_cancellation(
        self,
        task_id: str,
        *,
        reason: str = "",
        requested_by: str = "",
        requested_at: float | None = None,
    ) -> bool:
        now = requested_at or time.time()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET status='cancel_requested', cancel_requested_at=?, cancel_reason=?,
                    cancel_requested_by=?, updated_at=?
                WHERE task_id=? AND status='running'
                """,
                (now, str(reason or ""), str(requested_by or ""), now, task_id),
            )
        return cur.rowcount == 1

    def complete_task_cancellation(
        self,
        task_id: str,
        *,
        result: dict[str, Any] | None = None,
        run_id: str = "",
    ) -> bool:
        now = time.time()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET status='cancelled', cancelled_at=?, result_json=?,
                    run_id=COALESCE(NULLIF(?, ''), run_id), updated_at=?, lease_expires_at=NULL
                WHERE task_id=? AND status='cancel_requested'
                """,
                (now, json_dumps(result or {}), run_id, now, task_id),
            )
        return cur.rowcount == 1

    def attach_task_run(self, task_id: str, run_id: str) -> bool:
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET run_id=?, updated_at=?
                WHERE task_id=? AND status IN ('running', 'cancel_requested')
                  AND (run_id='' OR run_id=?)
                """,
                (run_id, time.time(), task_id, run_id),
            )
        return cur.rowcount == 1

    def claim_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_seconds: float,
        claimed_at: float | None = None,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = claimed_at or time.time()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET status='running', worker_id=?, heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE task_id=? AND status='queued'
                """,
                (str(worker_id), now, now + lease_seconds, now, task_id),
            )
        return cur.rowcount == 1

    def renew_task_lease(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_seconds: float,
        heartbeat_at: float | None = None,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = heartbeat_at or time.time()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE task_id=? AND worker_id=? AND status IN ('running', 'cancel_requested')
                """,
                (now, now + lease_seconds, now, task_id, str(worker_id)),
            )
        return cur.rowcount == 1

    def list_stale_tasks(self, *, now: float | None = None, limit: int = 100) -> list[dict[str, Any]]:
        cutoff = now or time.time()
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM yieldmind_tasks
                WHERE status IN ('running', 'cancel_requested')
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                ORDER BY lease_expires_at ASC
                LIMIT ?
                """,
                (cutoff, max(1, min(int(limit), 500))),
            ).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def interrupt_stale_task(
        self,
        task_id: str,
        *,
        reason: str,
        requested_by: str,
        confirmed_worker_stopped: bool,
        interrupted_at: float | None = None,
    ) -> bool:
        if not confirmed_worker_stopped:
            raise ValueError("confirmed_worker_stopped=true is required before interrupting a stale task")
        now = interrupted_at or time.time()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE yieldmind_tasks
                SET status='interrupted', interrupted_at=?, recovery_reason=?,
                    recovery_requested_by=?, lease_expires_at=NULL, updated_at=?
                WHERE task_id=? AND status IN ('running', 'cancel_requested')
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (now, str(reason or ""), str(requested_by or ""), now, task_id, now),
            )
        return cur.rowcount == 1
