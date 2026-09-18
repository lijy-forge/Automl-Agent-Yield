"""Session and memory primitives for multi-turn YieldMind interactions."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from yieldmind.database import YieldMindStore, connect, json_dumps, json_loads


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


class CreateSessionRequest(BaseModel):
    constraints: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class AddMessageRequest(BaseModel):
    session_id: str
    content: str
    role: str = "user"
    idempotency_key: str | None = None
    max_context_tokens: int = Field(default=1200, ge=200, le=8000)


class UpsertMemoryRequest(BaseModel):
    session_id: str = ""
    scope: str = "session"
    kind: str = "preference"
    content: str
    source_ref: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeleteMemoryRequest(BaseModel):
    memory_id: str


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
                    (session_id, status, created_at, updated_at, constraints_json, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, "active", now, now, json_dumps(request.constraints), request.summary),
            )
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            row = conn.execute("SELECT * FROM yieldmind_sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["constraints"] = json_loads(payload.pop("constraints_json", "{}"))
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
        sections = [
            ("fixed_instructions", "Use current explicit constraints before older memories. Do not trigger training unless requested."),
            ("current_constraints", json_dumps(session.get("constraints", {}))),
            ("memories", "\n".join(f"- {m['content']} [{m['memory_id']}]" for m in memories if m.get("status") == "active")),
            ("recent_messages", "\n".join(f"{m['role']}: {m['content']}" for m in messages)),
        ]
        selected: list[dict[str, Any]] = []
        used = 0
        clipped: list[str] = []
        for name, text in sections:
            tokens = _estimate_tokens(text)
            if used + tokens <= max_tokens or name in {"fixed_instructions", "current_constraints"}:
                selected.append({"name": name, "text": text, "estimated_tokens": tokens})
                used += tokens
            else:
                clipped.append(name)
        return {
            "session_id": session_id,
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
                return {"turn": payload, "idempotent": True, "context": self.build_context(request.session_id, max_tokens=request.max_context_tokens)}

        constraints, changes = self._extract_constraints(request.content, session.get("constraints", {}))
        action = "create_run" if any(word in request.content for word in ("运行", "再跑", "重新跑", "run again")) else "answer_only"
        if "为什么" in request.content or "why" in request.content.lower():
            action = "answer_only"
        turn_id = _new_id("turn")
        now = time.time()
        metadata = {"constraint_changes": changes, "action": action}
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
                    (turn_id, session_id, idempotency_key, role, content, status, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (turn_id, request.session_id, idempotency_key, request.role, request.content, "processed", now, json_dumps(metadata)),
            )
            conn.execute(
                """
                UPDATE yieldmind_sessions
                SET updated_at=?, constraints_json=?, current_run_id=COALESCE(NULLIF(?, ''), current_run_id)
                WHERE session_id=?
                """,
                (now, json_dumps(constraints), new_run_id, request.session_id),
            )
        turn = {
            "turn_id": turn_id,
            "session_id": request.session_id,
            "idempotency_key": idempotency_key,
            "role": request.role,
            "content": request.content,
            "status": "processed",
            "created_at": now,
            "metadata": metadata,
        }
        return {
            "turn": turn,
            "idempotent": False,
            "action": action,
            "run_id": new_run_id,
            "context": self.build_context(request.session_id, max_tokens=request.max_context_tokens),
        }

    def upsert_memory(self, request: UpsertMemoryRequest) -> dict[str, Any]:
        memory_id = _new_id("mem")
        now = time.time()
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_memories
                    (memory_id, session_id, scope, kind, content, source_ref, status,
                     created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    request.session_id,
                    request.scope,
                    request.kind,
                    request.content,
                    request.source_ref,
                    "active",
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
        return payload

    def search_memories(self, *, session_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
        with connect(self.store.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM yieldmind_memories
                WHERE status='active' AND (session_id='' OR session_id=?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, max(1, min(int(limit), 200))),
            ).fetchall()
        out = []
        for row in rows:
            payload = dict(row)
            payload["metadata"] = json_loads(payload.pop("metadata_json", "{}"))
            out.append(payload)
        return out

    def delete_memory(self, request: DeleteMemoryRequest) -> dict[str, Any]:
        now = time.time()
        with connect(self.store.db_path) as conn:
            cur = conn.execute(
                "UPDATE yieldmind_memories SET status='deleted', updated_at=? WHERE memory_id=?",
                (now, request.memory_id),
            )
        return {"ok": cur.rowcount > 0, "memory_id": request.memory_id, "status": "deleted" if cur.rowcount > 0 else "missing"}
