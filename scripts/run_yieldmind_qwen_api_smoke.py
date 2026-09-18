#!/usr/bin/env python3
"""Exercise FastAPI knowledge endpoints with the isolated Qwen3 runtime."""

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

from fastapi.testclient import TestClient

from yieldmind.database import json_dumps
from yieldmind.knowledge_base import (
    HttpEmbeddingFunction,
    QWEN3_INDEX_VERSION,
    qwen3_embedding_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-endpoint", required=True)
    parser.add_argument("--embedding-token-env", default="YIELDMIND_EMBEDDING_TOKEN")
    parser.add_argument("--sources-dir", default=str(PROJECT_ROOT / "knowledge_sources"))
    parser.add_argument("--query", default="如何选择屈服应力机理并防止数据泄漏")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "qwen_api_smoke"),
    )
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    os.environ["YIELDMIND_EMBEDDING_PROFILE"] = "qwen3-embedding-0.6b"
    os.environ["YIELDMIND_EMBEDDING_ENDPOINT"] = args.embedding_endpoint
    os.environ["YIELDMIND_EMBEDDING_TOKEN_ENV"] = args.embedding_token_env
    os.environ["YIELDMIND_CHROMA_DIR"] = str(run_root / "chroma")
    os.environ["YIELDMIND_CHROMA_COLLECTION"] = "yieldmind_qwen_api_smoke"

    from yieldmind.api import app

    profile = qwen3_embedding_profile(http=True)
    observer = HttpEmbeddingFunction(
        profile,
        endpoint=args.embedding_endpoint,
        token=os.environ.get(args.embedding_token_env, ""),
    )
    service_before = observer.health()
    client = TestClient(app)
    readiness = client.get("/health/dependencies")
    config_response = client.get("/api/knowledge/config")

    source_paths = sorted(Path(args.sources_dir).glob("*.md"))
    ingest_response = client.post(
        "/api/knowledge/ingest",
        json={
            "paths": [str(path) for path in source_paths],
            "index_version": QWEN3_INDEX_VERSION,
            "chunk_size": 700,
            "chunk_overlap": 80,
        },
    )
    search_args = {
        "query": args.query,
        "top_k": args.top_k,
        "index_version": QWEN3_INDEX_VERSION,
        "retrieval_mode": "hybrid",
    }
    search_response = client.post("/api/knowledge/search", json=search_args)
    tool_response = client.post("/api/tools/search_knowledge/call", json={"args": search_args})
    service_after = observer.health()

    readiness_payload = readiness.json()
    config_payload = config_response.json()
    ingest_payload = ingest_response.json()
    search_payload = search_response.json()
    tool_payload = tool_response.json()
    failed_documents = [
        item for item in ingest_payload.get("documents", []) if item.get("status") != "available"
    ]
    checks = {
        "readiness_ok": readiness.status_code == 200 and readiness_payload.get("ok") is True,
        "knowledge_dependency_ok": readiness_payload.get("knowledge", {}).get("ok") is True,
        "qwen_config_active": config_payload.get("profile") == "qwen3-embedding-0.6b",
        "ingest_ok": ingest_response.status_code == 200 and bool(ingest_payload.get("documents")) and not failed_documents,
        "direct_search_ok": search_response.status_code == 200 and bool(search_payload.get("hits")),
        "direct_search_uses_qwen": search_payload.get("embedding_model") == profile.model_id,
        "tool_search_ok": tool_response.status_code == 200 and tool_payload.get("ok") is True,
        "tool_search_uses_qwen": (tool_payload.get("result") or {}).get("embedding_model") == profile.model_id,
        "profile_fingerprint_matches": (
            search_payload.get("embedding_profile_fingerprint") == profile.fingerprint()
            and (tool_payload.get("result") or {}).get("embedding_profile_fingerprint") == profile.fingerprint()
        ),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "database_backend": readiness_payload.get("database", {}).get("backend", ""),
        "redis": readiness_payload.get("redis", {}),
        "knowledge_runtime": readiness_payload.get("knowledge", {}),
        "embedding_profile": profile.model_dump(mode="json"),
        "real_embedding_encode_calls": int(service_after.get("encode_calls", 0))
        - int(service_before.get("encode_calls", 0)),
        "real_embedding_encoded_texts": int(service_after.get("encoded_texts", 0))
        - int(service_before.get("encoded_texts", 0)),
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "ingestion": {
            "documents": len(ingest_payload.get("documents", [])),
            "chunks": sum(int(item.get("chunk_count", 0)) for item in ingest_payload.get("documents", [])),
        },
        "direct_search": {
            "hit_count": len(search_payload.get("hits", [])),
            "latency_ms": search_payload.get("latency_ms"),
        },
        "tool_search": {
            "hit_count": len((tool_payload.get("result") or {}).get("hits", [])),
            "latency_ms": (tool_payload.get("result") or {}).get("latency_ms"),
        },
        "limitations": [
            "TestClient exercises the real FastAPI route functions without a separate HTTP server process.",
            "The smoke validates real Qwen embeddings and tool routing, not LLM tool selection.",
        ],
    }
    output_path = run_root / "qwen_api_smoke.json"
    output_path.write_text(json_dumps(summary) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(output_path), **summary}, ensure_ascii=False, default=str))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
