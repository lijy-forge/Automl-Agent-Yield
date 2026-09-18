"""Session and memory primitives for multi-turn YieldMind interactions."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from yieldmind.database import YieldMindStore, connect, json_dumps, json_loads


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


class CreateSessionRequest(BaseModel):
    workspace_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    constraints: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class AddMessageRequest(BaseModel):
    session_id: str
    content: str
    role: str = "user"
    idempotency_key: str | None = None
    max_context_tokens: int = Field(default=1200, ge=200, le=8000)
    expected_constraint_version: int | None = Field(default=None, ge=0)


class UpsertMemoryRequest(BaseModel):
    session_id: str = ""
    workspace_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    scope: Literal["session", "workspace"] = "session"
    kind: str = "preference"
    content: str
    source_ref: str = Field(min_length=1)
    source_run_id: str = ""
    validation_status: Literal["candidate", "confirmed", "rejected"] = "candidate"
    applicability: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeleteMemoryRequest(BaseModel):
    memory_id: str


class ReviewMemoryRequest(BaseModel):
    memory_id: str
    validation_status: Literal["confirmed", "rejected"]


class UpdateSessionSummaryRequest(BaseModel):
    session_id: str
    summary: str
    through_turn_id: str


class MemoryVersionConflict(RuntimeError):
    """Raised when a session constraint update loses an optimistic-lock race."""


class SessionMemoryStore:
    def __init__(self, store: YieldMindStore | None = None) -> None:
        self.store = store or YieldMindStore()

    def create_session(self, request: CreateSessionRequest) -> dict[str, Any]:
        session_id = _new_id("ses")
        now = time.time()
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_sessions
                    (session_id, status, created_at, updated_at, workspace_id,
                     constraints_json, constraint_version, summary, summary_through_turn_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    "active",
                    now,
                    now,
                    request.workspace_id,
                    json_dumps(request.constraints),
                    0,
                    request.summary,
                    "",
                ),
            )
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            row = conn.execute("SELECT * FROM yieldmind_sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["constraints"] = json_loads(payload.pop("constraints_json", "{}"))
        payload["constraint_version"] = int(payload.get("constraint_version") or 0)
        return payload

    def list_messages(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with connect(self.store.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM yieldmind_turns
                WHERE session_id=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, max(1, min(int(limit), 200))),
            ).fetchall()
        out = []
        for row in reversed(rows):
            payload = dict(row)
            payload["metadata"] = json_loads(payload.pop("metadata_json", "{}"))
            payload["constraint_version"] = int(payload.get("constraint_version") or 0)
            out.append(payload)
        return out

    def _extract_constraints(self, content: str, current: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        updated = dict(current)
        changes: list[str] = []
        target_match = re.search(r"(?:目标列|target(?: column)?)\s*(?:改成|为|=|:)?\s*([A-Za-z_][\w]*)", content, re.I)
        if target_match:
            updated["target_column"] = target_match.group(1)
            changes.append(f"target_column={target_match.group(1)}")
        if "预算调小" in content or "搜索预算调小" in content or "smaller budget" in content.lower():
            updated["search_budget"] = "small"
            changes.append("search_budget=small")
        if "预算调大" in content or "larger budget" in content.lower():
            updated["search_budget"] = "large"
            changes.append("search_budget=large")
        if "不要重新训练" in content or "不重新训练" in content:
            updated["allow_training"] = False
            changes.append("allow_training=false")
        return updated, changes

    def build_context(self, session_id: str, *, max_tokens: int = 1200) -> dict[str, Any]:
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Unknown session_id: {session_id}")
        messages = self.list_messages(session_id, limit=30)
        memories = self.search_memories(session_id=session_id, limit=20)
        session_memories = [item for item in memories if item.get("scope") == "session"]
        workspace_memories = [item for item in memories if item.get("scope") == "workspace"]
        current_run = self.store.get_run(str(session.get("current_run_id") or ""))
        run_summary = (
            json_dumps(
                {
                    "run_id": current_run.get("run_id"),
                    "status": current_run.get("status"),
                    "mode": current_run.get("mode"),
                }
            )
            if current_run
            else ""
        )
        rolling_summary = str(session.get("summary") or "")
        if rolling_summary and session.get("summary_through_turn_id"):
            rolling_summary += f" [through_turn_id={session['summary_through_turn_id']}]"
        sections = [
            ("fixed_instructions", "Use current explicit constraints before older memories. Do not trigger training unless requested."),
            ("current_constraints", json_dumps(session.get("constraints", {}))),
            ("current_run", run_summary),
            (
                "session_memories",
                "\n".join(f"- {m['content']} [{m['memory_id']}]" for m in session_memories),
            ),
            (
                "workspace_memories",
                "\n".join(f"- {m['content']} [{m['memory_id']}]" for m in workspace_memories),
            ),
            ("rolling_summary", rolling_summary),
            ("recent_messages", "\n".join(f"{m['role']}: {m['content']}" for m in messages)),
        ]
        selected: list[dict[str, Any]] = []
        used = 0
        clipped: list[dict[str, str]] = []
        for name, text in sections:
            if not text:
                continue
            tokens = _estimate_tokens(text)
            if used + tokens <= max_tokens or name in {"fixed_instructions", "current_constraints"}:
                selected.append({"name": name, "text": text, "estimated_tokens": tokens})
                used += tokens
            else:
                clipped.append({"name": name, "reason": "context_token_budget_exceeded"})
        return {
            "session_id": session_id,
            "workspace_id": session.get("workspace_id", "default"),
            "constraint_version": int(session.get("constraint_version") or 0),
            "sections": selected,
            "estimated_tokens": used,
            "max_context_tokens": max_tokens,
            "clipped_sections": clipped,
        }

    def add_message(self, request: AddMessageRequest) -> dict[str, Any]:
        session = self.get_session(request.session_id)
        if not session:
            raise KeyError(f"Unknown session_id: {request.session_id}")
        idempotency_key = request.idempotency_key or _new_id("idem")
        with connect(self.store.db_path) as conn:
            existing = conn.execute(
                "SELECT * FROM yieldmind_turns WHERE session_id=? AND idempotency_key=?",
                (request.session_id, idempotency_key),
            ).fetchone()
            if existing:
                payload = dict(existing)
                payload["metadata"] = json_loads(payload.pop("metadata_json", "{}"))
                payload["constraint_version"] = int(payload.get("constraint_version") or 0)
                return {"turn": payload, "idempotent": True, "context": self.build_context(request.session_id, max_tokens=request.max_context_tokens)}

        constraints, changes = self._extract_constraints(request.content, session.get("constraints", {}))
        current_version = int(session.get("constraint_version") or 0)
        if request.expected_constraint_version is not None and request.expected_constraint_version != current_version:
            raise MemoryVersionConflict(
                f"Session constraint version changed: expected {request.expected_constraint_version}, current {current_version}."
            )
        next_version = current_version + 1 if changes else current_version
        now = time.time()
        if changes:
            with connect(self.store.db_path) as conn:
                updated = conn.execute(
                    """
                    UPDATE yieldmind_sessions
                    SET updated_at=?, constraints_json=?, constraint_version=?
                    WHERE session_id=? AND constraint_version=?
                    """,
                    (now, json_dumps(constraints), next_version, request.session_id, current_version),
                )
            if updated.rowcount != 1:
                latest = self.get_session(request.session_id) or {}
                raise MemoryVersionConflict(
                    f"Session constraint version changed during update: expected {current_version}, "
                    f"current {latest.get('constraint_version', 'unknown')}."
                )
        action = "create_run" if any(word in request.content for word in ("运行", "再跑", "重新跑", "run again")) else "answer_only"
        if "为什么" in request.content or "why" in request.content.lower():
            action = "answer_only"
        turn_id = _new_id("turn")
        metadata = {
            "constraint_changes": changes,
            "constraint_version_before": current_version,
            "constraint_version_after": next_version,
            "action": action,
        }
        parent_run_id = session.get("current_run_id") or ""
        new_run_id = ""
        if action == "create_run":
            new_run_id = self.store.create_run(
                mode="session_requested",
                source="session_memory",
                status="created",
                metadata={"session_id": request.session_id, "parent_run_id": parent_run_id, "constraints": constraints},
            )
            metadata["run_id"] = new_run_id
            metadata["parent_run_id"] = parent_run_id
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_turns
                    (turn_id, session_id, idempotency_key, role, content, status, created_at,
                     constraint_version, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    request.session_id,
                    idempotency_key,
                    request.role,
                    request.content,
                    "processed",
                    now,
                    next_version,
                    json_dumps(metadata),
                ),
            )
            conn.execute(
                """
                UPDATE yieldmind_sessions
                SET updated_at=?, current_run_id=COALESCE(NULLIF(?, ''), current_run_id)
                WHERE session_id=?
                """,
                (now, new_run_id, request.session_id),
            )
        turn = {
            "turn_id": turn_id,
            "session_id": request.session_id,
            "idempotency_key": idempotency_key,
            "role": request.role,
            "content": request.content,
            "status": "processed",
            "created_at": now,
            "constraint_version": next_version,
            "metadata": metadata,
        }
        return {
            "turn": turn,
            "idempotent": False,
            "action": action,
            "run_id": new_run_id,
            "constraint_version": next_version,
            "context": self.build_context(request.session_id, max_tokens=request.max_context_tokens),
        }

    def upsert_memory(self, request: UpsertMemoryRequest) -> dict[str, Any]:
        session = self.get_session(request.session_id) if request.session_id else None
        if request.scope == "session" and session is None:
            raise KeyError(f"Session-scoped memory requires a valid session_id: {request.session_id}")
        workspace_id = str((session or {}).get("workspace_id") or request.workspace_id)
        self._validate_memory_confirmation(
            kind=request.kind,
            validation_status=request.validation_status,
            source_run_id=request.source_run_id,
        )
        memory_id = _new_id("mem")
        now = time.time()
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_memories
                    (memory_id, session_id, workspace_id, scope, kind, content, source_ref, status,
                     validation_status, source_run_id, applicability_json,
                     created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    request.session_id if request.scope == "session" else "",
                    workspace_id,
                    request.scope,
                    request.kind,
                    request.content,
                    request.source_ref,
                    "active",
                    request.validation_status,
                    request.source_run_id,
                    json_dumps(request.applicability),
                    now,
                    now,
                    json_dumps(request.metadata),
                ),
            )
        return self.get_memory(memory_id) or {}

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            row = conn.execute("SELECT * FROM yieldmind_memories WHERE memory_id=?", (memory_id,)).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["metadata"] = json_loads(payload.pop("metadata_json", "{}"))
        payload["applicability"] = json_loads(payload.pop("applicability_json", "{}"))
        return payload

    def search_memories(
        self,
        *,
        session_id: str = "",
        workspace_id: str = "default",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: tuple[Any, ...]
        if session_id:
            session = self.get_session(session_id)
            if not session:
                raise KeyError(f"Unknown session_id: {session_id}")
            sql = """
                SELECT * FROM yieldmind_memories
                WHERE status='active' AND validation_status='confirmed'
                  AND ((scope='session' AND session_id=?) OR (scope='workspace' AND workspace_id=?))
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = (session_id, str(session.get("workspace_id") or "default"), max(1, min(int(limit), 200)))
        else:
            sql = """
                SELECT * FROM yieldmind_memories
                WHERE status='active' AND validation_status='confirmed'
                  AND scope='workspace' AND workspace_id=?
                ORDER BY updated_at DESC
                LIMIT ?
            """
            params = (workspace_id, max(1, min(int(limit), 200)))
        with connect(self.store.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for row in rows:
            payload = dict(row)
            payload["metadata"] = json_loads(payload.pop("metadata_json", "{}"))
            payload["applicability"] = json_loads(payload.pop("applicability_json", "{}"))
            out.append(payload)
        return out

    def review_memory(self, request: ReviewMemoryRequest) -> dict[str, Any]:
        memory = self.get_memory(request.memory_id)
        if not memory:
            raise KeyError(f"Unknown memory_id: {request.memory_id}")
        self._validate_memory_confirmation(
            kind=str(memory.get("kind") or ""),
            validation_status=request.validation_status,
            source_run_id=str(memory.get("source_run_id") or ""),
        )
        now = time.time()
        with connect(self.store.db_path) as conn:
            conn.execute(
                "UPDATE yieldmind_memories SET validation_status=?, updated_at=? WHERE memory_id=?",
                (request.validation_status, now, request.memory_id),
            )
        return self.get_memory(request.memory_id) or {}

    def update_summary(self, request: UpdateSessionSummaryRequest) -> dict[str, Any]:
        session = self.get_session(request.session_id)
        if not session:
            raise KeyError(f"Unknown session_id: {request.session_id}")
        with connect(self.store.db_path) as conn:
            turn = conn.execute(
                "SELECT turn_id FROM yieldmind_turns WHERE turn_id=? AND session_id=?",
                (request.through_turn_id, request.session_id),
            ).fetchone()
            if not turn:
                raise ValueError("Summary provenance turn does not belong to the session.")
            conn.execute(
                """
                UPDATE yieldmind_sessions
                SET summary=?, summary_through_turn_id=?, updated_at=?
                WHERE session_id=?
                """,
                (request.summary, request.through_turn_id, time.time(), request.session_id),
            )
        return self.get_session(request.session_id) or {}

    def _validate_memory_confirmation(self, *, kind: str, validation_status: str, source_run_id: str) -> None:
        if validation_status != "confirmed" or kind != "successful_experience":
            return
        if not source_run_id:
            raise ValueError("Confirmed successful_experience memory requires source_run_id.")
        run = self.store.get_run(source_run_id)
        if not run or run.get("status") != "passed":
            raise ValueError("Successful experience can only be confirmed from a passed run.")

    def delete_memory(self, request: DeleteMemoryRequest) -> dict[str, Any]:
        now = time.time()
        with connect(self.store.db_path) as conn:
            cur = conn.execute(
                "UPDATE yieldmind_memories SET status='deleted', updated_at=? WHERE memory_id=?",
                (now, request.memory_id),
            )
        return {"ok": cur.rowcount > 0, "memory_id": request.memory_id, "status": "deleted" if cur.rowcount > 0 else "missing"}
