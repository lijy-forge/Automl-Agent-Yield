#!/usr/bin/env python
"""Run reproducible vector, BM25, and hybrid retrieval baselines."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import PROJECT_ROOT, YieldMindStore, json_dumps
from yieldmind.knowledge_base import (
    DEFAULT_INDEX_VERSION,
    EmbeddingProfile,
    HttpEmbeddingFunction,
    KnowledgeBase,
    KnowledgeIngestRequest,
    QWEN3_INDEX_VERSION,
    QWEN3_MODEL_ID,
    QWEN3_QUERY_INSTRUCTION,
    QWEN3_REVISION,
    evaluate_retrieval,
)


DEFAULT_CASES = PROJECT_ROOT / "evals" / "yieldmind_retrieval_cases_v1.json"
DEFAULT_SOURCES = PROJECT_ROOT / "knowledge_sources"
DEFAULT_OUTPUT = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "retrieval_evals"
def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result = {"python": platform.python_version()}
    for package in ("chromadb", "langchain-text-splitters", "rank-bm25", "sentence-transformers", "transformers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _profile(args: argparse.Namespace) -> EmbeddingProfile:
    if args.embedding_provider == "local_hashing":
        return EmbeddingProfile(index_version=args.index_version or DEFAULT_INDEX_VERSION)
    if not args.model_id:
        raise ValueError("--model-id is required for sentence_transformers.")
    if not args.model_revision or args.model_revision.lower() == "main":
        raise ValueError("--model-revision must be an immutable model revision/commit, not 'main'.")
    provisional = f"yieldmind-st-{args.model_id}-{args.model_revision}-{args.dimensions}"
    index_version = args.index_version or provisional.replace("/", "-").replace(" ", "-")[:120]
    return EmbeddingProfile(
        provider="http_sentence_transformers" if args.embedding_endpoint else "sentence_transformers",
        model_id=args.model_id,
        revision=args.model_revision,
        dimensions=args.dimensions,
        normalize=True,
        metric="cosine",
        query_instruction=args.query_instruction,
        document_instruction=args.document_instruction,
        index_version=index_version,
    )


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "qwen3-embedding-0.6b":
        return
    args.embedding_provider = "sentence_transformers"
    args.model_id = QWEN3_MODEL_ID
    args.model_revision = QWEN3_REVISION
    args.dimensions = 1024
    args.query_instruction = QWEN3_QUERY_INSTRUCTION
    args.document_instruction = ""
    args.index_version = QWEN3_INDEX_VERSION
    if not args.hf_home:
        args.hf_home = str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "hf_cache")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    started = time.time()
    case_payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = case_payload["cases"]
    if len(cases) < 30:
        raise ValueError(f"Retrieval benchmark requires at least 30 cases; found {len(cases)}.")
    source_paths = sorted(Path(args.sources_dir).glob("*.md"))
    if not source_paths:
        raise ValueError(f"No Markdown sources found in {args.sources_dir}.")

    profile = _profile(args)
    embedding_service_health: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="yieldmind_retrieval_") as tmp:
        root = Path(tmp)
        store = YieldMindStore(root / "yieldmind.sqlite3")
        kb = KnowledgeBase(
            store=store,
            chroma_dir=root / "chroma",
            collection_name="yieldmind_retrieval_eval",
            embedding=(
                HttpEmbeddingFunction(
                    profile,
                    endpoint=args.embedding_endpoint,
                    token=os.environ.get(args.embedding_token_env, ""),
                    timeout_seconds=args.embedding_timeout_seconds,
                )
                if args.embedding_endpoint
                else None
            ),
            embedding_profile=profile,
            allow_model_download=args.allow_model_download,
        )
        ingest_started = time.perf_counter()
        ingest = kb.ingest(
            KnowledgeIngestRequest(
                paths=[str(path) for path in source_paths],
                index_version=profile.index_version,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )
        failures = [item for item in ingest["documents"] if item.get("status") != "available"]
        if failures:
            raise RuntimeError(f"Knowledge ingestion failed: {failures}")
        ingest_duration_seconds = time.perf_counter() - ingest_started
        baselines = {
            mode: evaluate_retrieval(kb, cases, top_k=args.top_k, retrieval_mode=mode)
            for mode in ("vector", "bm25", "hybrid")
        }
        if isinstance(kb.embedding, HttpEmbeddingFunction):
            embedding_service_health = kb.embedding.health()

    report: dict[str, Any] = {
        "status": "passed",
        "benchmark_version": case_payload.get("version", ""),
        "label_scope": case_payload.get("label_scope", ""),
        "case_count": len(cases),
        "source_count": len(source_paths),
        "source_paths": [str(path.relative_to(PROJECT_ROOT)) for path in source_paths],
        "embedding_profile": profile.model_dump(mode="json"),
        "embedding_profile_fingerprint": profile.fingerprint(),
        "embedding_execution_mode": (
            "offline_deterministic_hashing"
            if profile.provider == "local_hashing"
            else "isolated_http_sentence_transformer_real_inference"
            if profile.provider == "http_sentence_transformers"
            else "local_sentence_transformer_real_inference"
        ),
        "model_download_allowed": bool(args.allow_model_download),
        "hf_home": os.environ.get("HF_HOME", ""),
        "embedding_endpoint": args.embedding_endpoint,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "embedding_encode_calls": {
            "document_batches": int(getattr(kb.embedding, "document_encode_calls", 0)),
            "query_calls": int(getattr(kb.embedding, "query_encode_calls", 0)),
        },
        "embedding_service_health": embedding_service_health,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "baselines": baselines,
        "ingestion": {
            "document_count": len(ingest["documents"]),
            "chunk_count": sum(int(item.get("chunk_count", 0)) for item in ingest["documents"]),
            "split_version": ingest["split_version"],
            "chunk_size": ingest["chunk_size"],
            "chunk_overlap": ingest["chunk_overlap"],
            "collection": ingest["collection"],
            "duration_seconds": round(ingest_duration_seconds, 4),
        },
        "versions": _versions(),
        "duration_seconds": round(time.time() - started, 4),
        "limitations": [
            "Labels cover repository behavior and project domain constraints, not an external literature benchmark.",
            (
                "The default vector baseline uses deterministic hashing and is not a semantic embedding quality claim."
                if profile.provider == "local_hashing"
                else "This run uses real semantic embedding inference; the production default remains unchanged pending a larger independently reviewed benchmark."
            ),
            "No reranker is included; add one only after a measured baseline warrants the extra latency and dependencies.",
        ],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"retrieval_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json_dumps(report) + "\n", encoding="utf-8")
    return report, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("none", "qwen3-embedding-0.6b"), default="none")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--sources-dir", default=str(DEFAULT_SOURCES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--embedding-provider", choices=("local_hashing", "sentence_transformers"), default="local_hashing")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--model-revision", default="", help="Required immutable revision/commit for model profiles.")
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--query-instruction", default="")
    parser.add_argument("--document-instruction", default="")
    parser.add_argument("--index-version", default="")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--hf-home", default="", help="Optional isolated Hugging Face cache directory.")
    parser.add_argument("--embedding-endpoint", default="", help="Loopback endpoint for an isolated embedding service.")
    parser.add_argument("--embedding-token-env", default="YIELDMIND_EMBEDDING_TOKEN")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    _apply_preset(args)
    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser().resolve())
    try:
        report, output_path = run(args)
    except Exception as exc:
        print(json_dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    summary = {
        mode: {
            key: values[key]
            for key in ("recall_at_k", "mrr", "citation_accuracy_at_1", "mean_latency_ms", "max_latency_ms")
        }
        for mode, values in report["baselines"].items()
    }
    print(json_dumps({"status": report["status"], "report_path": str(output_path), "baselines": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
