"""Verified cross-run execution repair memories.

Failures are stored as candidates for audit, but only a failure followed by a
successful Operation result and an accepted Manager review becomes retrievable
guidance. This prevents an unverified workaround from poisoning future runs.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any, Callable

from yieldmind.database import YieldMindStore, connect, json_dumps, json_loads
from yieldmind.knowledge_base import EmbeddingFunction, HashingEmbeddingFunction
from yieldmind.memory import SessionMemoryStore, UpsertMemoryRequest


REPAIR_MEMORY_KIND = "verified_repair"
REPAIR_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
DEFAULT_REPAIR_MEMORY_MAX_AGE_DAYS = 90.0

DEPENDENCY_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scikit-learn", ("scikit-learn", "scikit learn", "sklearn")),
    ("numpy", ("numpy", "np.")),
    ("pandas", ("pandas", "pd.")),
    ("xgboost", ("xgboost", "xgb")),
    ("lightgbm", ("lightgbm", "lgbm")),
    ("torch", ("pytorch", "torch")),
)


def classify_error(text: str) -> str:
    normalized = normalized_error_signature(text)
    if any(token in normalized for token in ("nan", "dtype", "column", "feature", "shape", "keyerror", "target")):
        return "data_schema"
    if any(token in normalized for token in ("importerror", "modulenotfound", "no module named", "version", "attributeerror")):
        return "dependency"
    if any(token in normalized for token in ("timeout", "timed out", "out of memory", "oom", "killed", "resource")):
        return "resource"
    if any(token in normalized for token in ("artifact", "metrics.json", "predictions", "model file", "contract")):
        return "artifact_contract"
    return "execution"


def dependency_family(text: str) -> str:
    normalized = str(text or "").lower()
    for family, aliases in DEPENDENCY_FAMILY_PATTERNS:
        if any(alias in normalized for alias in aliases):
            return family
    return ""


def _clip(value: Any, limit: int = 3000) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def normalized_error_signature(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"(?:[a-z]:)?[/\\][^\s:]+", " <path> ", normalized)
    normalized = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:1200]


def operation_error_text(operation_result: dict[str, Any]) -> str:
    logs = [str(item) for item in operation_result.get("error_logs", []) or [] if str(item).strip()]
    action = str(operation_result.get("action_result") or "").strip()
    return "\n".join([*logs[-3:], action]).strip()


class RepairMemoryStore:
    def __init__(
        self,
        store: YieldMindStore,
        *,
        embedding_factory: Callable[[], EmbeddingFunction] | None = None,
        max_age_days: float = DEFAULT_REPAIR_MEMORY_MAX_AGE_DAYS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_age_days <= 0:
            raise ValueError("max_age_days must be greater than zero.")
        self.store = store
        self.memories = SessionMemoryStore(store)
        self.embedding_factory = embedding_factory or (lambda: HashingEmbeddingFunction())
        self.max_age_days = float(max_age_days)
        self.clock = clock
        self._embedding: EmbeddingFunction | None = None
        self._vector_cache: dict[tuple[str, float], list[float]] = {}
        self.last_retrieval_audit: dict[str, Any] = {}

    def _active_embedding(self) -> EmbeddingFunction:
        if self._embedding is None:
            self._embedding = self.embedding_factory()
        return self._embedding

    def _freshness(self, memory: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(memory.get("metadata") or {})
        source_field = "last_verified_at" if metadata.get("last_verified_at") else "verified_at"
        verified_at = float(metadata.get(source_field) or 0.0)
        if not verified_at:
            source_field = "updated_at_legacy_fallback"
            verified_at = float(memory.get("updated_at") or 0.0)
        age_seconds = max(0.0, self.clock() - verified_at) if verified_at else float("inf")
        age_days = age_seconds / 86400.0
        return {
            "fresh": age_days <= self.max_age_days,
            "verified_at": verified_at,
            "timestamp_source": source_field,
            "age_days": round(age_days, 6) if math.isfinite(age_days) else None,
            "max_age_days": self.max_age_days,
        }

    @staticmethod
    def _memory_search_text(memory: dict[str, Any]) -> str:
        metadata = dict(memory.get("metadata") or {})
        return "\n".join(
            value
            for value in (
                str(metadata.get("error_signature") or ""),
                str(metadata.get("failure") or ""),
                str(metadata.get("repair_summary") or ""),
                str(memory.get("content") or ""),
            )
            if value
        )

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    @staticmethod
    def _lexical_similarity(query: str, document: str) -> float:
        query_tokens = set(REPAIR_TOKEN_RE.findall(normalized_error_signature(query)))
        document_tokens = set(REPAIR_TOKEN_RE.findall(normalized_error_signature(document)))
        if not query_tokens or not document_tokens:
            return 0.0
        overlap = len(query_tokens & document_tokens)
        return overlap / math.sqrt(len(query_tokens) * len(document_tokens))

    @staticmethod
    def _major_minor(value: Any) -> str:
        match = re.match(r"^(\d+)\.(\d+)", str(value or ""))
        return f"{match.group(1)}.{match.group(2)}" if match else ""

    @classmethod
    def _applicability_match(
        cls,
        memory: dict[str, Any],
        current: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        saved = dict(memory.get("applicability") or {})
        category = str((memory.get("metadata") or {}).get("error_category") or "execution")
        checked: list[str] = []
        missing: list[str] = []
        mismatches: list[dict[str, str]] = []

        def compare(field: str, *, normalize: Callable[[Any], str] = lambda value: str(value or "")) -> None:
            expected = normalize(saved.get(field))
            actual = normalize(current.get(field))
            if not expected or not actual:
                if actual and not expected:
                    missing.append(field)
                return
            checked.append(field)
            if expected != actual:
                mismatches.append({"field": field, "memory": expected, "current": actual})

        if category == "data_schema":
            for field in ("dataset_schema", "dataset_role", "target_column", "feature_signature"):
                compare(field)
        elif category == "dependency":
            compare("python_version", normalize=cls._major_minor)
            saved_dependencies = dict(saved.get("dependency_versions") or {})
            current_dependencies = dict(current.get("dependency_versions") or {})
            for name in sorted(set(saved_dependencies) & set(current_dependencies)):
                expected = cls._major_minor(saved_dependencies[name])
                actual = cls._major_minor(current_dependencies[name])
                if expected and actual:
                    checked.append(f"dependency_versions.{name}")
                    if expected != actual:
                        mismatches.append(
                            {"field": f"dependency_versions.{name}", "memory": expected, "current": actual}
                        )
            if current_dependencies and not saved_dependencies:
                missing.append("dependency_versions")
        elif category == "resource":
            compare("execution_backend")
            compare("platform_system")
            compare("platform_machine")
        elif category == "artifact_contract":
            compare("operation_contract_version")

        score = 1.0 if checked and not missing else 0.7 if not mismatches else 0.0
        return not mismatches, {
            "compatible": not mismatches,
            "error_category": category,
            "score": score,
            "checked_fields": checked,
            "missing_memory_fields": missing,
            "mismatches": mismatches,
            "legacy_unscoped": not checked and bool(current),
        }

    def _rank(
        self,
        memories: list[dict[str, Any]],
        *,
        query_text: str,
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            ranked = []
            for index, memory in enumerate(memories):
                item = dict(memory)
                item["retrieval"] = {
                    "rank": index + 1,
                    "score": round(1.0 / (index + 1), 6),
                    "embedding_score": 0.0,
                    "lexical_score": 0.0,
                    "recency_score": round(1.0 / (index + 1), 6),
                    "method": "recency_fallback",
                    "query_signature": "",
                }
                ranked.append(item)
            return ranked

        documents = [self._memory_search_text(memory) for memory in memories]
        embedding_scores = [0.0] * len(memories)
        method = "lexical_fallback"
        degraded_error = ""
        try:
            embedding = self._active_embedding()
            query_vector = embedding.embed_query(query_text)
            missing_indices: list[int] = []
            missing_documents: list[str] = []
            document_vectors: list[list[float] | None] = [None] * len(memories)
            for index, memory in enumerate(memories):
                cache_key = (str(memory.get("memory_id") or ""), float(memory.get("updated_at") or 0.0))
                cached = self._vector_cache.get(cache_key)
                if cached is None:
                    missing_indices.append(index)
                    missing_documents.append(documents[index])
                else:
                    document_vectors[index] = cached
            if missing_documents:
                encoded = embedding.embed_documents(missing_documents)
                for index, vector in zip(missing_indices, encoded):
                    memory = memories[index]
                    cache_key = (str(memory.get("memory_id") or ""), float(memory.get("updated_at") or 0.0))
                    self._vector_cache[cache_key] = vector
                    document_vectors[index] = vector
            embedding_scores = [
                max(0.0, self._cosine(query_vector, vector or []))
                for vector in document_vectors
            ]
            method = (
                "semantic_embedding"
                if type(embedding).__name__ != "HashingEmbeddingFunction"
                else "deterministic_hashing_embedding"
            )
        except Exception as exc:
            degraded_error = type(exc).__name__

        query_signature = normalized_error_signature(query_text)
        scored: list[tuple[float, dict[str, Any]]] = []
        for index, (memory, document, embedding_score) in enumerate(zip(memories, documents, embedding_scores)):
            lexical_score = self._lexical_similarity(query_text, document)
            recency_score = 1.0 / (index + 1)
            memory_signature = str((memory.get("metadata") or {}).get("error_signature") or "")
            signature_match = 1.0 if memory_signature and memory_signature == query_signature else 0.0
            applicability = dict(memory.get("applicability_match") or {})
            applicability_score = float(applicability.get("score", 0.7))
            raw_score = 0.65 * embedding_score + 0.25 * lexical_score + 0.05 * recency_score + 0.05 * signature_match
            score = raw_score * (0.8 + 0.2 * applicability_score)
            item = dict(memory)
            item["retrieval"] = {
                "rank": 0,
                "score": round(score, 6),
                "embedding_score": round(embedding_score, 6),
                "lexical_score": round(lexical_score, 6),
                "recency_score": round(recency_score, 6),
                "signature_match": bool(signature_match),
                "applicability_score": round(applicability_score, 6),
                "method": method,
                "query_signature": query_signature,
                "degraded_error": degraded_error,
            }
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], float(pair[1].get("updated_at") or 0.0)), reverse=True)
        ranked = [item for _, item in scored]
        for rank, item in enumerate(ranked, start=1):
            item["retrieval"]["rank"] = rank
        return ranked

    @staticmethod
    def _fingerprint(execution_mode: str, error_text: str) -> str:
        signature = normalized_error_signature(error_text)
        return hashlib.sha256(f"{execution_mode}:{signature}".encode("utf-8")).hexdigest()[:24]

    def _by_source_ref(self, source_ref: str) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            row = conn.execute(
                "SELECT memory_id FROM yieldmind_memories WHERE source_ref=? ORDER BY updated_at DESC LIMIT 1",
                (source_ref,),
            ).fetchone()
        return self.memories.get_memory(str(row["memory_id"])) if row else None

    def _confirmed_by_fingerprint(
        self,
        *,
        workspace_id: str,
        execution_mode: str,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            rows = conn.execute(
                """
                SELECT memory_id FROM yieldmind_memories
                WHERE status='active' AND validation_status='confirmed'
                  AND scope='workspace' AND workspace_id=? AND kind=?
                ORDER BY updated_at DESC
                LIMIT 200
                """,
                (workspace_id, REPAIR_MEMORY_KIND),
            ).fetchall()
        for row in rows:
            memory = self.memories.get_memory(str(row["memory_id"]))
            if not memory:
                continue
            mode = str((memory.get("applicability") or {}).get("execution_mode") or "")
            saved_fingerprint = str((memory.get("metadata") or {}).get("fingerprint") or "")
            if mode == execution_mode and saved_fingerprint == fingerprint:
                return memory
        return None

    def record_candidate(
        self,
        *,
        workspace_id: str,
        run_id: str,
        execution_mode: str,
        operation_result: dict[str, Any],
        applicability_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        error_text = operation_error_text(operation_result)
        if not error_text:
            return None
        signature = normalized_error_signature(error_text)
        error_category = classify_error(error_text)
        fingerprint = self._fingerprint(execution_mode, error_text)
        if self._confirmed_by_fingerprint(
            workspace_id=workspace_id,
            execution_mode=execution_mode,
            fingerprint=fingerprint,
        ):
            return None
        source_ref = f"run:{run_id}:repair:{fingerprint}"
        existing = self._by_source_ref(source_ref)
        if existing:
            return existing
        return self.memories.upsert_memory(
            UpsertMemoryRequest(
                workspace_id=workspace_id,
                scope="workspace",
                kind=REPAIR_MEMORY_KIND,
                content=(
                    "Unverified execution failure. Do not use as cross-run guidance until a later "
                    f"Operation succeeds and Manager accepts it. Failure: {_clip(error_text)}"
                ),
                source_ref=source_ref,
                source_run_id=run_id,
                validation_status="candidate",
                applicability={
                    "execution_mode": execution_mode,
                    "workflow_kind": "domain",
                    **dict(applicability_context or {}),
                },
                metadata={
                    "error_signature": signature,
                    "fingerprint": fingerprint,
                    "failure": _clip(error_text),
                    "error_category": error_category,
                    "operation_rcode": operation_result.get("rcode"),
                    "created_from": "operation_error",
                },
            )
        )

    def confirm(
        self,
        memory_id: str,
        *,
        successful_operation: dict[str, Any],
        manager_review: dict[str, Any],
        repair_summary: str = "",
    ) -> dict[str, Any] | None:
        memory = self.memories.get_memory(memory_id)
        if not memory or memory.get("kind") != REPAIR_MEMORY_KIND:
            return None
        if int(successful_operation.get("rcode", 1)) != 0 or not manager_review.get("passed"):
            return memory
        metadata = dict(memory.get("metadata") or {})
        outcome = _clip(successful_operation.get("action_result") or "Operation completed successfully.")
        guidance = _clip(repair_summary or "Use the verified bounded retry/fallback and rerun deterministic guardrails.", 1800)
        verified_now = self.clock()
        content = (
            "Verified execution repair.\n"
            f"Execution mode: {(memory.get('applicability') or {}).get('execution_mode', '')}.\n"
            f"Failure signature: {metadata.get('error_signature', '')}.\n"
            f"Observed failure: {metadata.get('failure', '')}.\n"
            f"Verified successful outcome: {outcome}.\n"
            f"Prevention/repair guidance: {guidance}.\n"
            "Before generating or executing similar code, check this failure condition and the relevant contract."
        )
        metadata.update(
            {
                "verified_outcome": outcome,
                "repair_summary": guidance,
                "manager_decision": manager_review.get("decision"),
                "verified_at": verified_now,
                "last_verified_at": verified_now,
                "verification_count": 1,
                "verification_history": [
                    {
                        "run_id": str(memory.get("source_run_id") or ""),
                        "verified_at": verified_now,
                        "source": "initial_workflow_verification",
                        "manager_decision": manager_review.get("decision"),
                    }
                ],
            }
        )
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                UPDATE yieldmind_memories
                SET content=?, validation_status='confirmed', updated_at=?, metadata_json=?
                WHERE memory_id=? AND status='active'
                """,
                (content, verified_now, json_dumps(metadata), memory_id),
            )
        return self.memories.get_memory(memory_id)

    def reconfirm(
        self,
        memory_id: str,
        *,
        evidence_run_id: str,
        successful_operation: dict[str, Any],
        manager_review: dict[str, Any],
        source: str = "workflow_reuse",
        note: str = "",
    ) -> dict[str, Any] | None:
        memory = self.memories.get_memory(memory_id)
        if (
            not memory
            or memory.get("kind") != REPAIR_MEMORY_KIND
            or memory.get("status") != "active"
            or memory.get("validation_status") != "confirmed"
        ):
            return None
        if int(successful_operation.get("rcode", 1)) != 0 or not manager_review.get("passed"):
            return None
        metadata = dict(memory.get("metadata") or {})
        history = [dict(item) for item in (metadata.get("verification_history") or []) if isinstance(item, dict)]
        if evidence_run_id and any(str(item.get("run_id") or "") == evidence_run_id for item in history):
            return memory
        verified_now = self.clock()
        history.append(
            {
                "run_id": evidence_run_id,
                "verified_at": verified_now,
                "source": source,
                "manager_decision": manager_review.get("decision"),
                "outcome": _clip(successful_operation.get("action_result") or "Successful reuse verified.", 600),
                "note": _clip(note, 600),
            }
        )
        metadata.update(
            {
                "last_verified_at": verified_now,
                "verification_count": max(int(metadata.get("verification_count") or 1) + 1, len(history)),
                "verification_history": history[-20:],
                "last_reconfirmation_source": source,
            }
        )
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                UPDATE yieldmind_memories
                SET updated_at=?, metadata_json=?
                WHERE memory_id=? AND status='active' AND validation_status='confirmed'
                """,
                (verified_now, json_dumps(metadata), memory_id),
            )
        return self.memories.get_memory(memory_id)

    def reconfirm_from_passed_run(
        self,
        memory_id: str,
        *,
        evidence_run_id: str,
        reviewer: str,
        note: str = "",
    ) -> dict[str, Any] | None:
        memory = self.memories.get_memory(memory_id)
        if (
            not memory
            or memory.get("kind") != REPAIR_MEMORY_KIND
            or memory.get("status") != "active"
            or memory.get("validation_status") != "confirmed"
        ):
            return None
        run = self.store.get_run(evidence_run_id)
        if not run:
            raise KeyError(f"Unknown evidence_run_id: {evidence_run_id}")
        if run.get("status") != "passed":
            raise ValueError("Repair memory reconfirmation requires a passed evidence run.")
        if run.get("source") != "yield_domain_stategraph":
            raise ValueError("Repair memory reconfirmation requires a domain StateGraph evidence run.")
        memory_workspace = str(memory.get("workspace_id") or "default")
        run_workspace = str((run.get("metadata") or {}).get("workspace_id") or "")
        if not run_workspace or run_workspace != memory_workspace:
            raise ValueError("Evidence run workspace_id does not match repair memory workspace_id.")
        memory_mode = str((memory.get("applicability") or {}).get("execution_mode") or "")
        run_mode = str((run.get("result") or {}).get("execution_mode") or "")
        if memory_mode and run_mode and memory_mode != run_mode:
            raise ValueError("Evidence run execution_mode does not match repair memory execution_mode.")
        return self.reconfirm(
            memory_id,
            evidence_run_id=evidence_run_id,
            successful_operation={"rcode": 0, "action_result": "The supplied evidence run finished with status=passed."},
            manager_review={"passed": True, "decision": "explicit_reconfirmation"},
            source="explicit_api_reconfirmation",
            note=f"reviewer={reviewer}; {note}".strip("; "),
        )

    def _reuse_attempt(self, reuse_id: str) -> dict[str, Any] | None:
        with connect(self.store.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM yieldmind_repair_memory_reuse_attempts WHERE reuse_id=?",
                (reuse_id,),
            ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["metadata"] = json_loads(payload.pop("metadata_json", "{}"))
        if payload.get("manager_passed") is not None:
            payload["manager_passed"] = bool(payload["manager_passed"])
        return payload

    def begin_reuse_attempt(
        self,
        memory_id: str,
        *,
        run_id: str,
        workspace_id: str,
        execution_mode: str,
        matched_round: int,
        applied_round: int,
        retrieval: dict[str, Any] | None = None,
        query_signature: str = "",
    ) -> dict[str, Any]:
        """Persist a reuse only when a matched repair reaches a revision round."""

        memory = self.memories.get_memory(memory_id)
        if (
            not memory
            or memory.get("kind") != REPAIR_MEMORY_KIND
            or memory.get("status") != "active"
            or memory.get("validation_status") != "confirmed"
        ):
            raise ValueError("Reuse attempt requires an active confirmed repair memory.")
        if str(memory.get("workspace_id") or "default") != workspace_id:
            raise ValueError("Reuse attempt workspace_id does not match repair memory workspace_id.")
        memory_mode = str((memory.get("applicability") or {}).get("execution_mode") or "")
        if memory_mode and memory_mode != execution_mode:
            raise ValueError("Reuse attempt execution_mode does not match repair memory execution_mode.")
        run = self.store.get_run(run_id)
        if not run:
            raise KeyError(f"Unknown reuse run_id: {run_id}")
        run_workspace = str((run.get("metadata") or {}).get("workspace_id") or "")
        if run_workspace and run_workspace != workspace_id:
            raise ValueError("Reuse run workspace_id does not match repair memory workspace_id.")
        if applied_round <= matched_round:
            raise ValueError("applied_round must be later than matched_round.")

        reuse_key = f"{memory_id}:{run_id}:{applied_round}"
        reuse_id = f"reuse_{hashlib.sha256(reuse_key.encode('utf-8')).hexdigest()[:20]}"
        now = self.clock()
        metadata = {
            "query_signature": _clip(query_signature, 1200),
            "retrieval": dict(retrieval or {}),
            "evidence_policy": "error_match_then_revision",
        }
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT INTO yieldmind_repair_memory_reuse_attempts
                    (reuse_id, memory_id, run_id, workspace_id, execution_mode,
                     matched_round, applied_round, status, matched_at, applied_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'applied', ?, ?, ?)
                ON CONFLICT(memory_id, run_id, applied_round) DO NOTHING
                """,
                (
                    reuse_id,
                    memory_id,
                    run_id,
                    workspace_id,
                    execution_mode,
                    int(matched_round),
                    int(applied_round),
                    now,
                    now,
                    json_dumps(metadata),
                ),
            )
            row = conn.execute(
                """
                SELECT reuse_id FROM yieldmind_repair_memory_reuse_attempts
                WHERE memory_id=? AND run_id=? AND applied_round=?
                """,
                (memory_id, run_id, int(applied_round)),
            ).fetchone()
        attempt = self._reuse_attempt(str(row["reuse_id"])) if row else None
        if not attempt:
            raise RuntimeError("Failed to persist Repair Memory reuse attempt.")
        return attempt

    def complete_reuse_attempt(
        self,
        reuse_id: str,
        *,
        operation_result: dict[str, Any],
        manager_review: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Finalize an applied reuse once its revised Operation is reviewed."""

        attempt = self._reuse_attempt(reuse_id)
        if not attempt:
            return None
        if attempt.get("status") in {"succeeded", "failed"}:
            return attempt
        try:
            rcode = int(operation_result.get("rcode", 1))
        except (TypeError, ValueError):
            rcode = 1
        manager_passed = bool(manager_review.get("passed"))
        status = "succeeded" if rcode == 0 and manager_passed else "failed"
        metadata = dict(attempt.get("metadata") or {})
        metadata["outcome"] = {
            "action_result": _clip(operation_result.get("action_result"), 1000),
            "error": _clip(operation_error_text(operation_result), 1200),
            "manager_feedback": _clip(manager_review.get("feedback"), 1000),
        }
        completed_at = self.clock()
        with connect(self.store.db_path) as conn:
            conn.execute(
                """
                UPDATE yieldmind_repair_memory_reuse_attempts
                SET status=?, operation_rcode=?, manager_passed=?, manager_decision=?,
                    completed_at=?, metadata_json=?
                WHERE reuse_id=? AND status='applied'
                """,
                (
                    status,
                    rcode,
                    int(manager_passed),
                    _clip(manager_review.get("decision"), 500),
                    completed_at,
                    json_dumps(metadata),
                    reuse_id,
                ),
            )
        return self._reuse_attempt(reuse_id)

    def list_reuse_attempts(
        self,
        *,
        workspace_id: str = "",
        memory_id: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if workspace_id:
            clauses.append("workspace_id=?")
            params.append(workspace_id)
        if memory_id:
            clauses.append("memory_id=?")
            params.append(memory_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        with connect(self.store.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT reuse_id FROM yieldmind_repair_memory_reuse_attempts
                {where}
                ORDER BY applied_at DESC, reuse_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [attempt for row in rows if (attempt := self._reuse_attempt(str(row["reuse_id"]))) is not None]

    def reuse_outcome_summary(
        self,
        *,
        workspace_id: str = "",
        minimum_claimable_observations: int = 30,
    ) -> dict[str, Any]:
        attempts = self.list_reuse_attempts(workspace_id=workspace_id, limit=1000)
        observed = [item for item in attempts if item.get("status") in {"succeeded", "failed"}]
        episodes: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for item in observed:
            key = (str(item.get("run_id") or ""), int(item.get("applied_round") or 0))
            episodes.setdefault(key, []).append(item)
        successful_episode_count = sum(
            all(item.get("status") == "succeeded" for item in items)
            for items in episodes.values()
        )
        failed_episode_count = len(episodes) - successful_episode_count
        by_memory: dict[str, dict[str, Any]] = {}
        for item in observed:
            memory_id = str(item.get("memory_id") or "")
            group = by_memory.setdefault(
                memory_id,
                {"memory_id": memory_id, "observed_count": 0, "success_count": 0, "failure_count": 0},
            )
            group["observed_count"] += 1
            group["success_count"] += int(item.get("status") == "succeeded")
            group["failure_count"] += int(item.get("status") == "failed")
        for group in by_memory.values():
            group["success_rate"] = round(group["success_count"] / group["observed_count"], 6)
        observed_count = len(episodes)
        minimum = max(1, int(minimum_claimable_observations))
        return {
            "workspace_id": workspace_id,
            "attempt_count": len(attempts),
            "open_attempt_count": sum(item.get("status") == "applied" for item in attempts),
            "observed_count": observed_count,
            "observed_attempt_count": len(observed),
            "success_count": successful_episode_count,
            "failure_count": failed_episode_count,
            "success_rate": (
                round(successful_episode_count / observed_count, 6) if observed_count else None
            ),
            "computable": observed_count > 0,
            "claimable": observed_count >= minimum,
            "minimum_claimable_observations": minimum,
            "evidence_policy": "error_match_then_revision",
            "aggregation_unit": "run_revision_episode",
            "by_memory": sorted(by_memory.values(), key=lambda item: item["memory_id"]),
        }

    def retrieve(
        self,
        *,
        workspace_id: str,
        execution_mode: str,
        limit: int = 5,
        query_text: str = "",
        min_relevance: float = 0.0,
        applicability_context: dict[str, Any] | None = None,
        strict_error_match: bool = False,
    ) -> list[dict[str, Any]]:
        with connect(self.store.db_path) as conn:
            rows = conn.execute(
                """
                SELECT memory_id FROM yieldmind_memories
                WHERE status='active' AND validation_status='confirmed'
                  AND scope='workspace' AND workspace_id=? AND kind=?
                ORDER BY updated_at DESC
                LIMIT 200
                """,
                (workspace_id, REPAIR_MEMORY_KIND),
            ).fetchall()
        memories = [
            memory
            for row in rows
            if (memory := self.memories.get_memory(str(row["memory_id"]))) is not None
        ]
        matched = []
        rejected: list[dict[str, Any]] = []
        current_applicability = dict(applicability_context or {})
        query_category = classify_error(query_text) if strict_error_match and query_text.strip() else ""
        query_dependency_family = dependency_family(query_text) if query_category == "dependency" else ""
        for memory in memories:
            if memory.get("kind") != REPAIR_MEMORY_KIND:
                continue
            applicable_mode = str((memory.get("applicability") or {}).get("execution_mode") or "")
            if applicable_mode and applicable_mode != execution_mode:
                continue
            freshness = self._freshness(memory)
            if not freshness["fresh"]:
                rejected.append(
                    {
                        "memory_id": str(memory.get("memory_id") or ""),
                        "reason": "verification_expired",
                        "freshness": freshness,
                        "error_category": str((memory.get("metadata") or {}).get("error_category") or "execution"),
                        "mismatches": [],
                    }
                )
                continue
            memory_metadata = dict(memory.get("metadata") or {})
            memory_category = str(memory_metadata.get("error_category") or "execution")
            if query_category and memory_category != query_category:
                rejected.append(
                    {
                        "memory_id": str(memory.get("memory_id") or ""),
                        "reason": "error_category_mismatch",
                        "error_category": memory_category,
                        "query_error_category": query_category,
                        "mismatches": [],
                    }
                )
                continue
            memory_dependency_family = (
                dependency_family(str(memory_metadata.get("failure") or ""))
                if memory_category == "dependency"
                else ""
            )
            if (
                query_dependency_family
                and memory_dependency_family
                and query_dependency_family != memory_dependency_family
            ):
                rejected.append(
                    {
                        "memory_id": str(memory.get("memory_id") or ""),
                        "reason": "dependency_family_mismatch",
                        "error_category": memory_category,
                        "query_dependency_family": query_dependency_family,
                        "memory_dependency_family": memory_dependency_family,
                        "mismatches": [],
                    }
                )
                continue
            compatible, applicability = self._applicability_match(memory, current_applicability)
            memory = dict(memory)
            memory["applicability_match"] = applicability
            memory["freshness"] = freshness
            if not compatible:
                rejected.append(
                    {
                        "memory_id": str(memory.get("memory_id") or ""),
                        "reason": "applicability_mismatch",
                        "error_category": applicability.get("error_category"),
                        "mismatches": applicability.get("mismatches") or [],
                    }
                )
                continue
            matched.append(memory)
        ranked = self._rank(matched, query_text=query_text)
        if query_text.strip() and min_relevance > 0:
            ranked = [
                memory
                for memory in ranked
                if float((memory.get("retrieval") or {}).get("score") or 0.0) >= min_relevance
            ]
        selected = ranked[: max(1, min(int(limit), 20))]
        self.last_retrieval_audit = {
            "candidate_count": len(memories),
            "applicable_count": len(matched),
            "rejected_count": len(rejected),
            "rejected": rejected[:20],
            "selected_count": len(selected),
            "query_error_category": query_category,
            "query_dependency_family": query_dependency_family,
            "strict_error_match": strict_error_match,
        }
        return selected

    @staticmethod
    def prompt_context(memories: list[dict[str, Any]], *, max_chars: int = 6000) -> str:
        if not memories:
            return ""
        lines = [
            "Verified cross-run repair memories. Treat these as preflight checks, not as permission to bypass current contracts:"
        ]
        used = len(lines[0])
        for memory in memories:
            retrieval = dict(memory.get("retrieval") or {})
            relevance = (
                f" relevance={float(retrieval.get('score') or 0.0):.3f}"
                f" via {retrieval.get('method')}"
                if retrieval
                else ""
            )
            line = f"- [{memory.get('memory_id')}{relevance}] {_clip(memory.get('content'), 2200)}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)
