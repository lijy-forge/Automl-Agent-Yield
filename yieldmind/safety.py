"""Safety helpers for YieldMind tool calls, evidence, and prompt budgets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from yieldmind.database import YieldMindStore, connect
from yieldmind.knowledge_base import DEFAULT_INDEX_VERSION


REDACTION = "[REDACTED]"
SENSITIVE_KEY_RE = re.compile(
    r"(authorization|api[_-]?key|access[_-]?key|password|passwd|token|secret|private[_-]?key|"
    r"connection[_-]?string|database[_-]?url)",
    re.IGNORECASE,
)
STRING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{8,}", re.IGNORECASE)),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b")),
    ("database_url", re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb|redis)://[^\s'\"<>]+", re.IGNORECASE)),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")),
)


class RedactionFinding(BaseModel):
    path: str
    kind: str
    reason: str


class RedactRequest(BaseModel):
    payload: Any
    replacement: str = REDACTION


class RedactedPayload(BaseModel):
    payload: Any
    redaction_count: int
    findings: list[RedactionFinding] = Field(default_factory=list)


class BudgetSection(BaseModel):
    name: str
    text: str = ""
    tokens: int | None = Field(default=None, ge=0)
    priority: int = Field(default=50, ge=0, le=100)
    required: bool = False


class BudgetRequest(BaseModel):
    sections: list[BudgetSection] = Field(..., min_length=1)
    max_input_tokens: int = Field(..., ge=1)
    reserved_output_tokens: int = Field(default=512, ge=0)
    min_section_tokens: int = Field(default=32, ge=1)
    include_text: bool = True


class BudgetPlan(BaseModel):
    estimator: str
    max_input_tokens: int
    reserved_output_tokens: int
    available_input_tokens: int
    selected_sections: list[dict[str, Any]]
    dropped_sections: list[dict[str, Any]]
    total_selected_tokens: int
    over_budget: bool
    reasons: list[str] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    chunk_id: str
    text_hash: str = ""
    index_version: str = DEFAULT_INDEX_VERSION
    document_id: str = ""
    document_version: str = ""


class EvidenceValidationRequest(BaseModel):
    refs: list[EvidenceRef] = Field(..., min_length=1)
    require_text_hash: bool = True
    require_index_version: bool = True


class EvidenceValidationResult(BaseModel):
    ok: bool
    valid_refs: list[dict[str, Any]]
    invalid_refs: list[dict[str, Any]]
    checked_count: int
    valid_count: int
    invalid_count: int


def _child_path(path: str, key: str) -> str:
    if key.startswith("["):
        return f"{path}{key}"
    return f"{path}.{key}" if path else key


def _redact_string(value: str, path: str, replacement: str) -> tuple[str, list[RedactionFinding], int]:
    findings: list[RedactionFinding] = []
    redacted = value
    count = 0
    for kind, pattern in STRING_PATTERNS:
        redacted, substitutions = pattern.subn(replacement, redacted)
        if substitutions:
            count += substitutions
            findings.append(
                RedactionFinding(path=path, kind=kind, reason=f"Matched {substitutions} sensitive string pattern(s).")
            )
    return redacted, findings, count


def redact_payload(request: RedactRequest) -> RedactedPayload:
    findings: list[RedactionFinding] = []

    def visit(value: Any, path: str) -> tuple[Any, int]:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            count = 0
            for key, item in value.items():
                key_str = str(key)
                next_path = _child_path(path, key_str)
                if SENSITIVE_KEY_RE.search(key_str):
                    out[key_str] = request.replacement
                    count += 1
                    findings.append(
                        RedactionFinding(path=next_path, kind="sensitive_key", reason="Matched sensitive field name.")
                    )
                else:
                    out[key_str], child_count = visit(item, next_path)
                    count += child_count
            return out, count
        if isinstance(value, list):
            out_list = []
            count = 0
            for idx, item in enumerate(value):
                redacted, child_count = visit(item, _child_path(path, f"[{idx}]"))
                out_list.append(redacted)
                count += child_count
            return out_list, count
        if isinstance(value, tuple):
            out_tuple = []
            count = 0
            for idx, item in enumerate(value):
                redacted, child_count = visit(item, _child_path(path, f"[{idx}]"))
                out_tuple.append(redacted)
                count += child_count
            return out_tuple, count
        if isinstance(value, str):
            redacted, string_findings, string_count = _redact_string(value, path, request.replacement)
            findings.extend(string_findings)
            return redacted, string_count
        return value, 0

    payload, count = visit(request.payload, "$")
    return RedactedPayload(payload=payload, redaction_count=count, findings=findings)


def estimate_tokens(text: str) -> tuple[int, str]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text or "")), "tiktoken_cl100k_base"
    except Exception:
        return max(1, (len(text or "") + 3) // 4), "chars_div_4"


@dataclass(frozen=True)
class _SectionCost:
    section: BudgetSection
    tokens: int
    estimator: str


def plan_token_budget(request: BudgetRequest) -> BudgetPlan:
    costs: list[_SectionCost] = []
    estimator = "provided_or_chars_div_4"
    for section in request.sections:
        if section.tokens is None:
            token_count, section_estimator = estimate_tokens(section.text)
        else:
            token_count, section_estimator = int(section.tokens), "provided"
        if section_estimator != "provided":
            estimator = section_estimator
        costs.append(_SectionCost(section=section, tokens=max(0, token_count), estimator=section_estimator))

    available = max(0, request.max_input_tokens - request.reserved_output_tokens)
    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    reasons: list[str] = []
    total = 0

    required = [item for item in costs if item.section.required]
    optional = [item for item in costs if not item.section.required]
    optional.sort(key=lambda item: (-item.section.priority, item.section.name))

    def section_payload(item: _SectionCost, *, included_tokens: int, clipped: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": item.section.name,
            "required": item.section.required,
            "priority": item.section.priority,
            "estimated_tokens": item.tokens,
            "included_tokens": included_tokens,
            "clipped": clipped,
        }
        if request.include_text:
            if clipped and item.tokens:
                ratio = max(0.0, min(1.0, included_tokens / item.tokens))
                keep_chars = max(0, int(len(item.section.text) * ratio))
                payload["text"] = item.section.text[:keep_chars]
            else:
                payload["text"] = item.section.text
        return payload

    for item in required:
        if total + item.tokens <= available:
            selected.append(section_payload(item, included_tokens=item.tokens))
            total += item.tokens
        elif total < available and available - total >= request.min_section_tokens:
            included = available - total
            selected.append(section_payload(item, included_tokens=included, clipped=True))
            total += included
            reasons.append(f"Required section {item.section.name!r} was clipped to fit the budget.")
        else:
            selected.append(section_payload(item, included_tokens=0, clipped=True))
            reasons.append(f"Required section {item.section.name!r} could not fit inside the available budget.")

    for item in optional:
        remaining = available - total
        if item.tokens <= remaining:
            selected.append(section_payload(item, included_tokens=item.tokens))
            total += item.tokens
        elif remaining >= request.min_section_tokens:
            selected.append(section_payload(item, included_tokens=remaining, clipped=True))
            total += remaining
            reasons.append(f"Optional section {item.section.name!r} was clipped to use remaining budget.")
        else:
            dropped.append(
                {
                    "name": item.section.name,
                    "required": item.section.required,
                    "priority": item.section.priority,
                    "estimated_tokens": item.tokens,
                    "reason": "insufficient remaining budget",
                }
            )

    over_budget = any(item.get("required") and item.get("included_tokens", 0) < item.get("estimated_tokens", 0) for item in selected)
    return BudgetPlan(
        estimator=estimator,
        max_input_tokens=request.max_input_tokens,
        reserved_output_tokens=request.reserved_output_tokens,
        available_input_tokens=available,
        selected_sections=selected,
        dropped_sections=dropped,
        total_selected_tokens=total,
        over_budget=over_budget,
        reasons=reasons,
    )


def validate_evidence_refs(store: YieldMindStore, request: EvidenceValidationRequest) -> EvidenceValidationResult:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    with connect(store.db_path) as conn:
        for ref in request.refs:
            row = conn.execute(
                "SELECT * FROM yieldmind_document_chunks WHERE chunk_id=?",
                (ref.chunk_id,),
            ).fetchone()
            if row is None:
                invalid.append({"ref": ref.model_dump(), "reason": "chunk_id not found"})
                continue
            payload = dict(row)
            reasons = []
            if ref.document_id and ref.document_id != payload.get("document_id"):
                reasons.append("document_id mismatch")
            if ref.document_version and ref.document_version != payload.get("document_version"):
                reasons.append("document_version mismatch")
            if request.require_text_hash and ref.text_hash and ref.text_hash != payload.get("text_hash"):
                reasons.append("text_hash mismatch")
            if request.require_text_hash and not ref.text_hash:
                reasons.append("missing text_hash")
            if request.require_index_version and ref.index_version and ref.index_version != payload.get("index_version"):
                reasons.append("index_version mismatch")
            if request.require_index_version and not ref.index_version:
                reasons.append("missing index_version")
            if reasons:
                invalid.append({"ref": ref.model_dump(), "reason": "; ".join(reasons)})
                continue
            valid.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "document_version": payload.get("document_version"),
                    "index_version": payload.get("index_version"),
                    "text_hash": payload.get("text_hash"),
                    "source_path": payload.get("source_path"),
                    "title": payload.get("title"),
                    "text_preview": str(payload.get("text") or "")[:240],
                }
            )
    return EvidenceValidationResult(
        ok=not invalid,
        valid_refs=valid,
        invalid_refs=invalid,
        checked_count=len(request.refs),
        valid_count=len(valid),
        invalid_count=len(invalid),
    )
