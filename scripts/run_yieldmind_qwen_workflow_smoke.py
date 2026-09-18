#!/usr/bin/env python3
"""Run the deterministic StateGraph with real Qwen3 retrieval evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import YieldMindStore, connect, json_dumps
from yieldmind.knowledge_base import (
    HttpEmbeddingFunction,
    KnowledgeBase,
    KnowledgeIngestRequest,
    QWEN3_INDEX_VERSION,
    qwen3_embedding_profile,
)
from yieldmind.tools import registry_for_workspace
from yieldmind.task_queue import redis_health
from yieldmind.workflow import WorkflowRequest, run_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-endpoint", required=True)
    parser.add_argument("--embedding-token-env", default="YIELDMIND_EMBEDDING_TOKEN")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--database-url", default=os.environ.get("YIELDMIND_DATABASE_URL", ""))
    parser.add_argument("--require-redis", action="store_true")
    parser.add_argument("--sources-dir", default=str(PROJECT_ROOT / "knowledge_sources"))
    parser.add_argument(
        "--query",
        default="屈服应力机理选择、数据泄漏防护、候选模型评测和执行隔离需要哪些证据",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--n-splits", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "qwen_workflow_smoke"),
    )
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    store = (
        YieldMindStore(database_url=args.database_url, initialize=False)
        if args.database_url
        else YieldMindStore(run_root / "yieldmind.sqlite3")
    )
    profile = qwen3_embedding_profile(http=True)
    embedding = HttpEmbeddingFunction(
        profile,
        endpoint=args.embedding_endpoint,
        token=os.environ.get(args.embedding_token_env, ""),
        timeout_seconds=args.embedding_timeout_seconds,
    )
    service_before = embedding.health()
    knowledge_base = KnowledgeBase(
        store=store,
        chroma_dir=run_root / "chroma",
        collection_name="yieldmind_qwen_workflow_smoke",
        embedding=embedding,
        embedding_profile=profile,
    )
    source_paths = sorted(Path(args.sources_dir).glob("*.md"))
    if not source_paths:
        raise ValueError(f"No Markdown sources found in {args.sources_dir}.")
    ingest_started = time.perf_counter()
    ingest = knowledge_base.ingest(
        KnowledgeIngestRequest(
            paths=[str(path) for path in source_paths],
            index_version=QWEN3_INDEX_VERSION,
            chunk_size=700,
            chunk_overlap=80,
        )
    )
    ingest_seconds = time.perf_counter() - ingest_started
    failed_documents = [item for item in ingest["documents"] if item.get("status") != "available"]
    if failed_documents:
        raise RuntimeError(f"Knowledge ingestion failed: {failed_documents}")

    registry = registry_for_workspace(store=store, knowledge_base=knowledge_base)
    workflow_started = time.perf_counter()
    result = run_workflow(
        WorkflowRequest(
            prompt="Build and evaluate a yield-stress model with validated domain evidence.",
            n_samples=args.n_samples,
            n_splits=args.n_splits,
            use_knowledge=True,
            knowledge_query=args.query,
            knowledge_top_k=args.top_k,
            knowledge_index_version=QWEN3_INDEX_VERSION,
            knowledge_retrieval_mode="hybrid",
        ),
        store=store,
        registry=registry,
    )
    workflow_seconds = time.perf_counter() - workflow_started
    service_after = embedding.health()
    thread_id = str(result.get("thread_id") or "")
    checkpoint_count = 0
    if store.backend == "postgresql" and thread_id:
        with connect(store.db_path) as conn:
            checkpoint_row = conn.execute(
                "SELECT COUNT(*) AS count FROM checkpoints WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
        checkpoint_count = int(checkpoint_row["count"]) if checkpoint_row else 0
    redis_status = redis_health()

    report_path = Path(result.get("artifacts", {}).get("report_json", ""))
    report_payload = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    stages = [stage.get("stage") for stage in result.get("stages", [])]
    tool_names = [item.get("tool") for item in result.get("tool_results", [])]
    checks = {
        "workflow_passed": result.get("status") == "passed",
        "langgraph_stategraph": result.get("workflow_backend") == "langgraph_stategraph",
        "retrieve_evidence_stage": "retrieve_evidence" in stages,
        "search_tool_called": "search_knowledge" in tool_names,
        "evidence_validation_called": "validate_evidence_refs" in tool_names,
        "evidence_refs_present": bool(result.get("evidence_refs")),
        "evidence_validation_passed": bool(result.get("evidence_summary", {}).get("validation_ok")),
        "report_contains_evidence": bool(report_payload.get("evidence_refs")),
        "qwen_profile_used": result.get("evidence_summary", {}).get("embedding_model") == profile.model_id,
    }
    if store.backend == "postgresql":
        checks["postgres_checkpoints_persisted"] = checkpoint_count > 0
    if args.require_redis:
        checks["redis_reachable"] = bool(redis_status.get("ok"))
    embedding_calls = int(service_after.get("encode_calls", 0)) - int(service_before.get("encode_calls", 0))
    encoded_texts = int(service_after.get("encoded_texts", 0)) - int(service_before.get("encoded_texts", 0))
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "database_backend": store.backend,
        "workflow_run_id": result.get("run_id", ""),
        "thread_id": thread_id,
        "workflow_backend": result.get("workflow_backend", ""),
        "checkpoint_backend": result.get("checkpoint_backend", ""),
        "checkpoint_count": checkpoint_count,
        "redis": redis_status,
        "embedding_profile": profile.model_dump(mode="json"),
        "embedding_service": {
            key: service_after.get(key)
            for key in (
                "model_id",
                "revision",
                "dimensions",
                "device",
                "versions",
                "process_peak_rss_bytes",
            )
        },
        "real_embedding_encode_calls": embedding_calls,
        "real_embedding_encoded_texts": encoded_texts,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "ingestion": {
            "documents": len(ingest["documents"]),
            "chunks": sum(int(item.get("chunk_count", 0)) for item in ingest["documents"]),
            "duration_seconds": round(ingest_seconds, 4),
        },
        "evidence_summary": result.get("evidence_summary", {}),
        "evidence_refs": result.get("evidence_refs", []),
        "stages": stages,
        "tool_names": tool_names,
        "workflow_duration_seconds": round(workflow_seconds, 4),
        "workflow_report_path": str(report_path),
        "limitations": [
            "The workflow uses deterministic modeling tools and real Qwen embeddings, but makes no LLM call.",
            "The knowledge sources are repository-maintained project documents, not an independently reviewed literature corpus.",
        ],
    }
    output_path = run_root / "qwen_workflow_smoke.json"
    output_path.write_text(json_dumps(summary) + "\n", encoding="utf-8")
    print(json.dumps({"smoke_report_path": str(output_path), **summary}, ensure_ascii=False, default=str))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
