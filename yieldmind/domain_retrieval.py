"""Knowledge-base retrieval adapter for the live domain workflow.

The core knowledge store returns rich chunk records, while the original domain
agents consume a compact ``snippets`` contract.  This module keeps the
translation deterministic and validates every primary evidence reference
before it can enter CandidateAgent or ModelAgent context.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from yieldmind.database import YieldMindStore
from yieldmind.knowledge_base import KnowledgeBase, KnowledgeSearchRequest
from yieldmind.safety import EvidenceValidationRequest, validate_evidence_refs


def _unique_nonempty(values: Iterable[Any], *, limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def build_domain_knowledge_queries(
    user_prompt: str,
    extra_queries: list[str] | None,
    data_profile: dict[str, Any],
    *,
    max_queries: int = 3,
) -> list[str]:
    """Build a small, auditable query set from the request and data capability."""
    columns = {
        str(column).strip().lower()
        for column in (data_profile.get("columns") or [])
        if str(column).strip()
    }
    derived: list[str] = []
    if columns.intersection({"phi", "phi_max", "sp_percent", "w_b", "fa_ratio"}):
        derived.append(
            "YODEL packing density phi_m superplasticizer formulation yield stress model"
        )
    if any("shear" in column for column in columns) or columns.intersection(
        {"gamma_dot", "shear_rate", "flow_curve"}
    ):
        derived.append(
            "Herschel Bulkley Bingham shear rate flow curve yield stress rheology"
        )
    if not derived:
        derived.append(
            "yield stress slurry machine learning rheology model validation data leakage"
        )
    return _unique_nonempty(
        [user_prompt, *(extra_queries or []), *derived],
        limit=max(1, int(max_queries)),
    )


def _hit_link(hit: dict[str, Any]) -> str:
    doi = str(hit.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    return str(hit.get("source_url") or hit.get("source_path") or "")


def _hit_to_snippet(
    hit: dict[str, Any],
    *,
    query: str,
    routing: dict[str, Any],
) -> dict[str, Any]:
    chunk_id = str(hit.get("chunk_id") or "")
    context = str(hit.get("context_text") or hit.get("text") or "").strip()
    return {
        "source_id": f"KB:{chunk_id}",
        "source": "yieldmind_knowledge_base",
        "provider": "yieldmind_knowledge_base",
        "source_type": f"knowledge_{hit.get('corpus') or 'unknown'}",
        "evidence_origin": "knowledge_base",
        "evidence_role": "retrieved_chunk",
        "category": hit.get("corpus") or "knowledge",
        "title": hit.get("title") or hit.get("section") or "YieldMind knowledge evidence",
        "link": _hit_link(hit),
        "snippet": context,
        "query": query,
        "retrieval_score": hit.get("score"),
        "retrieval_channels": hit.get("retrieval_channels") or [],
        "retrieval_channel_ranks": hit.get("retrieval_channel_ranks") or {},
        "chunk_id": chunk_id,
        "context_chunk_ids": hit.get("context_chunk_ids") or [chunk_id],
        "parent_id": hit.get("parent_id"),
        "text_hash": hit.get("text_hash"),
        "index_version": hit.get("index_version"),
        "document_id": hit.get("document_id"),
        "document_version": hit.get("document_version"),
        "source_path": hit.get("source_path"),
        "corpus": hit.get("corpus"),
        "section": hit.get("section"),
        "page_start": hit.get("page_start"),
        "page_end": hit.get("page_end"),
        "split_version": hit.get("split_version"),
        "routing": routing,
    }


def retrieve_domain_knowledge(
    knowledge_base: KnowledgeBase,
    store: YieldMindStore,
    *,
    queries: list[str],
    top_k: int = 5,
    context_budget_chars: int = 12000,
) -> dict[str, Any]:
    """Run routed hybrid retrieval and return validated, globally budgeted snippets."""
    started_queries: list[dict[str, Any]] = []
    candidates_by_query: list[list[tuple[dict[str, Any], str, dict[str, Any]]]] = []
    errors: list[str] = []
    index_version = knowledge_base.embedding_profile.index_version
    for query in _unique_nonempty(queries, limit=3):
        try:
            result = knowledge_base.search(
                KnowledgeSearchRequest(
                    query=query,
                    top_k=max(1, min(int(top_k), 20)),
                    index_version=index_version,
                    retrieval_mode="hybrid",
                    routing_mode="auto",
                    expand_parent=True,
                    parent_context_max_chars=min(4000, max(500, int(context_budget_chars))),
                    deduplicate_parents=True,
                    context_budget_chars=max(500, int(context_budget_chars)),
                )
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        routing = dict(result.get("routing") or {})
        hits = [hit for hit in (result.get("hits") or []) if isinstance(hit, dict)]
        started_queries.append(
            {
                "query": query,
                "routing": routing,
                "hit_count": len(hits),
                "latency_ms": result.get("latency_ms"),
                "selection_policy": result.get("selection_policy") or {},
            }
        )
        candidates_by_query.append([(hit, query, routing) for hit in hits])

    raw_candidates: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for rank in range(max((len(items) for items in candidates_by_query), default=0)):
        for items in candidates_by_query:
            if rank < len(items):
                raw_candidates.append(items[rank])

    deduplicated: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    seen_groups: set[str] = set()
    for hit, query, routing in raw_candidates:
        group_id = str(hit.get("parent_id") or hit.get("chunk_id") or "")
        if not group_id or group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        deduplicated.append((hit, query, routing))

    validation_payload = {
        "ok": True,
        "valid_refs": [],
        "invalid_refs": [],
        "checked_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
    }
    valid_chunk_ids: set[str] = set()
    if deduplicated:
        validation = validate_evidence_refs(
            store,
            EvidenceValidationRequest(
                refs=[
                    {
                        "chunk_id": str(hit.get("chunk_id") or ""),
                        "text_hash": str(hit.get("text_hash") or ""),
                        "index_version": str(hit.get("index_version") or ""),
                        "document_id": str(hit.get("document_id") or ""),
                        "document_version": str(hit.get("document_version") or ""),
                    }
                    for hit, _query, _routing in deduplicated
                ]
            ),
        )
        validation_payload = validation.model_dump(mode="json")
        valid_chunk_ids = {
            str(item.get("chunk_id") or "") for item in validation.valid_refs
        }

    snippets: list[dict[str, Any]] = []
    remaining = max(500, int(context_budget_chars))
    for hit, query, routing in deduplicated:
        if str(hit.get("chunk_id") or "") not in valid_chunk_ids:
            continue
        snippet = _hit_to_snippet(hit, query=query, routing=routing)
        text = str(snippet.get("snippet") or "")
        if not text or remaining <= 0:
            continue
        if len(text) > remaining:
            if snippets and remaining < 200:
                break
            snippet["snippet"] = text[:remaining]
            snippet["context_truncated"] = True
        else:
            snippet["context_truncated"] = False
        snippets.append(snippet)
        remaining -= len(str(snippet["snippet"]))
        if remaining <= 0:
            break

    corpus_counts = Counter(str(item.get("corpus") or "unknown") for item in snippets)
    status = "passed" if snippets and not errors else "degraded" if snippets else "failed" if errors else "empty"
    return {
        "status": status,
        "queries": started_queries,
        "requested_queries": _unique_nonempty(queries, limit=3),
        "retrieval_mode": "hybrid",
        "routing_mode": "auto",
        "top_k_per_query": max(1, min(int(top_k), 20)),
        "context_budget_chars": max(500, int(context_budget_chars)),
        "context_chars_used": sum(len(str(item.get("snippet") or "")) for item in snippets),
        "embedding_profile": knowledge_base.embedding_profile.model_dump(mode="json"),
        "embedding_profile_fingerprint": knowledge_base.embedding_profile.fingerprint(),
        "embedding_model": knowledge_base.embedding_profile.model_id,
        "provider_summary": {"yieldmind_knowledge_base": len(snippets)} if snippets else {},
        "source_type_summary": {
            f"knowledge_{corpus}": count for corpus, count in sorted(corpus_counts.items())
        },
        "evidence_validation": validation_payload,
        "selected_valid_count": len(snippets),
        "selected_chunk_ids": [str(item.get("chunk_id") or "") for item in snippets],
        "snippets": snippets,
        "errors": errors,
        "real_llm_calls": 0,
        "embedding_inference": knowledge_base.embedding_profile.provider != "local_hashing",
    }


def _round_robin(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            merged.append(left[index])
        if index < len(right):
            merged.append(right[index])
    return merged


def merge_domain_search_report(
    external_report: dict[str, Any],
    knowledge_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge knowledge and external evidence without erasing either provenance."""
    knowledge_report = knowledge_report or {}
    external = [item for item in (external_report.get("snippets") or []) if isinstance(item, dict)]
    knowledge = [item for item in (knowledge_report.get("snippets") or []) if isinstance(item, dict)]
    snippets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _round_robin(knowledge, external):
        identity = str(
            item.get("source_id")
            or item.get("chunk_id")
            or item.get("link")
            or f"{item.get('title')}:{item.get('snippet')}"
        )
        if identity in seen:
            continue
        seen.add(identity)
        snippets.append(item)

    provider_summary = Counter(external_report.get("provider_summary") or {})
    provider_summary.update(knowledge_report.get("provider_summary") or {})
    source_type_summary = Counter(external_report.get("source_type_summary") or {})
    source_type_summary.update(knowledge_report.get("source_type_summary") or {})
    source_quality = dict(external_report.get("source_quality") or {})
    source_quality["knowledge_evidence_validation"] = knowledge_report.get("evidence_validation") or {}
    if knowledge_report:
        source_quality["knowledge_status"] = knowledge_report.get("status")

    merged = dict(external_report)
    merged.update(
        {
            "snippets": snippets,
            "provider_summary": dict(provider_summary),
            "source_type_summary": dict(source_type_summary),
            "source_quality": source_quality,
            "knowledge_search": knowledge_report,
            "evidence_counts": {
                "knowledge": len(knowledge),
                "external": len(external),
                "combined": len(snippets),
            },
            "note": (
                f"Combined {len(knowledge)} validated knowledge-base snippets and "
                f"{len(external)} external-search snippets."
                if snippets
                else "No knowledge-base or external-search evidence was available."
            ),
        }
    )
    return merged
