#!/usr/bin/env python3
"""Validate a separately running YieldMind API over loopback HTTP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import json_dumps
from yieldmind.knowledge_base import (
    HttpEmbeddingFunction,
    QWEN3_INDEX_VERSION,
    qwen3_embedding_profile,
)

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def validate_loopback_http_url(value: str, *, label: str) -> str:
    normalized = value.rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError(f"{label} must use HTTP on a loopback host.")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError(f"{label} must be an origin URL without a path, query, or fragment.")
    return normalized


class LoopbackJsonClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 300.0) -> None:
        self.base_url = validate_loopback_http_url(base_url, label="API base URL")
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        data = json_dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return int(response.status), json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                payload_out = json.loads(body)
            except json.JSONDecodeError:
                payload_out = {"detail": body}
            return int(exc.code), payload_out
        except urllib.error.URLError as exc:
            raise RuntimeError(f"YieldMind API unavailable at {self.base_url}: {exc.reason}") from exc


def _write_report(output_dir: str, name: str, summary: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    output_path = run_root / name
    output_path.write_text(json_dumps(summary) + "\n", encoding="utf-8")
    return output_path


def _run_unready(args: argparse.Namespace, client: LoopbackJsonClient) -> dict[str, Any]:
    live_status, live = client.request("GET", "/health")
    dependency_status, dependencies = client.request("GET", "/health/dependencies")
    config_status, config = client.request("GET", "/api/knowledge/config")
    knowledge = dependencies.get("knowledge", {})
    checks = {
        "liveness_remains_available": live_status == 200 and live.get("ok") is True,
        "qwen_profile_selected": live.get("knowledge_profile") == "qwen3-embedding-0.6b",
        "readiness_rejects_dependency_failure": dependency_status == 503 and dependencies.get("ok") is False,
        "database_still_ready": dependencies.get("database", {}).get("ok") is True,
        "redis_still_ready": dependencies.get("redis", {}).get("ok") is True,
        "knowledge_is_not_ready": knowledge.get("ok") is False,
        "configured_runtime_is_visible": config_status == 200 and config.get("profile") == "qwen3-embedding-0.6b",
    }
    error_type = str(knowledge.get("error", "")).split(":", 1)[0]
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "scenario": "qwen_dependency_unavailable",
        "transport": "loopback_tcp_http",
        "http_statuses": {
            "liveness": live_status,
            "readiness": dependency_status,
            "knowledge_config": config_status,
        },
        "checks": checks,
        "knowledge_error_type": error_type,
        "real_embedding_encode_calls": 0,
        "real_embedding_encoded_texts": 0,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "limitations": [
            "This negative smoke validates dependency gating, not automatic failover or high availability.",
            "No embedding or LLM inference is expected in this scenario.",
        ],
    }


def _run_ready(args: argparse.Namespace, client: LoopbackJsonClient) -> dict[str, Any]:
    if not args.embedding_endpoint:
        raise ValueError("--embedding-endpoint is required for the ready scenario.")
    endpoint = validate_loopback_http_url(args.embedding_endpoint, label="Embedding endpoint")
    profile = qwen3_embedding_profile(http=True)
    observer = HttpEmbeddingFunction(
        profile,
        endpoint=endpoint,
        token=os.environ.get(args.embedding_token_env, ""),
        timeout_seconds=args.timeout_seconds,
    )
    service_before = observer.health()
    live_status, live = client.request("GET", "/health")
    dependency_status, dependencies = client.request("GET", "/health/dependencies")
    config_status, config = client.request("GET", "/api/knowledge/config")

    source_paths = sorted(Path(args.sources_dir).glob("*.md"))
    ingest_status, ingest = client.request(
        "POST",
        "/api/knowledge/ingest",
        {
            "paths": [str(path.resolve()) for path in source_paths],
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
    search_status, search = client.request("POST", "/api/knowledge/search", search_args)
    tool_status, tool = client.request(
        "POST",
        "/api/tools/search_knowledge/call",
        {"args": search_args},
    )
    service_after = observer.health()

    failed_documents = [item for item in ingest.get("documents", []) if item.get("status") != "available"]
    tool_result = tool.get("result") or {}
    fingerprint = profile.fingerprint()
    checks = {
        "liveness_ok": live_status == 200 and live.get("ok") is True,
        "readiness_ok": dependency_status == 200 and dependencies.get("ok") is True,
        "database_ready": dependencies.get("database", {}).get("ok") is True,
        "redis_ready": dependencies.get("redis", {}).get("ok") is True,
        "knowledge_ready": dependencies.get("knowledge", {}).get("ok") is True,
        "qwen_config_active": config_status == 200 and config.get("profile") == "qwen3-embedding-0.6b",
        "source_documents_found": bool(source_paths),
        "ingest_ok": ingest_status == 200 and bool(ingest.get("documents")) and not failed_documents,
        "direct_search_ok": search_status == 200 and bool(search.get("hits")),
        "direct_search_uses_qwen": search.get("embedding_model") == profile.model_id,
        "tool_search_ok": tool_status == 200 and tool.get("ok") is True and bool(tool_result.get("hits")),
        "tool_search_uses_qwen": tool_result.get("embedding_model") == profile.model_id,
        "profile_fingerprint_matches": (
            search.get("embedding_profile_fingerprint") == fingerprint
            and tool_result.get("embedding_profile_fingerprint") == fingerprint
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "scenario": "qwen_dependency_ready",
        "transport": "loopback_tcp_http",
        "http_statuses": {
            "liveness": live_status,
            "readiness": dependency_status,
            "knowledge_config": config_status,
            "ingest": ingest_status,
            "direct_search": search_status,
            "tool_search": tool_status,
        },
        "checks": checks,
        "database_backend": dependencies.get("database", {}).get("backend", ""),
        "redis_ok": dependencies.get("redis", {}).get("ok") is True,
        "embedding_profile": profile.model_dump(mode="json"),
        "real_embedding_encode_calls": int(service_after.get("encode_calls", 0))
        - int(service_before.get("encode_calls", 0)),
        "real_embedding_encoded_texts": int(service_after.get("encoded_texts", 0))
        - int(service_before.get("encoded_texts", 0)),
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "ingestion": {
            "documents": len(ingest.get("documents", [])),
            "chunks": sum(int(item.get("chunk_count", 0)) for item in ingest.get("documents", [])),
        },
        "direct_search": {
            "hit_count": len(search.get("hits", [])),
            "latency_ms": search.get("latency_ms"),
        },
        "tool_search": {
            "hit_count": len(tool_result.get("hits", [])),
            "latency_ms": tool_result.get("latency_ms"),
        },
        "limitations": [
            "This is one functional HTTP smoke run, not a throughput, concurrency, or production network benchmark.",
            "The smoke validates real Qwen embeddings and tool routing, not LLM tool selection.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8072")
    parser.add_argument("--embedding-endpoint", default="")
    parser.add_argument("--embedding-token-env", default="YIELDMIND_EMBEDDING_TOKEN")
    parser.add_argument("--sources-dir", default=str(PROJECT_ROOT / "knowledge_sources"))
    parser.add_argument("--query", default="如何选择屈服应力机理并防止数据泄漏")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--expect-unready", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "qwen_api_http"),
    )
    args = parser.parse_args()

    client = LoopbackJsonClient(args.api_base_url, timeout_seconds=args.timeout_seconds)
    summary = _run_unready(args, client) if args.expect_unready else _run_ready(args, client)
    name = "qwen_api_http_unready.json" if args.expect_unready else "qwen_api_http_ready.json"
    output_path = _write_report(args.output_dir, name, summary)
    report_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {"report_path": str(output_path), "report_sha256": report_sha256, **summary},
            ensure_ascii=False,
            default=str,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
