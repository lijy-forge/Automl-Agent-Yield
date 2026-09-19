#!/usr/bin/env python3
"""Ingest project rules and the verified literature corpus into one filterable index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import YieldMindStore
from yieldmind.knowledge_base import KnowledgeIngestRequest, KnowledgeSearchRequest
from yieldmind.knowledge_runtime import configured_knowledge_base
from yieldmind.literature import build_literature_ingest_request, load_literature_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "knowledge_sources" / "literature" / "manifest.json"),
    )
    parser.add_argument(
        "--document-dir",
        default=str(PROJECT_ROOT / "knowledge_sources" / "literature" / "documents"),
    )
    parser.add_argument("--literature-chunk-size", type=int, default=1800)
    parser.add_argument("--literature-chunk-overlap", type=int, default=180)
    args = parser.parse_args()

    store = YieldMindStore()
    knowledge_base = configured_knowledge_base(store)
    project_paths = sorted(
        str(path.resolve())
        for path in (PROJECT_ROOT / "knowledge_sources").glob("*.md")
        if path.is_file()
    )
    project_result = knowledge_base.ingest(
        KnowledgeIngestRequest(
            paths=project_paths,
            index_version=knowledge_base.embedding_profile.index_version,
            chunk_size=700,
            chunk_overlap=80,
        )
    )

    manifest = load_literature_manifest(args.manifest)
    literature_request = build_literature_ingest_request(
        manifest,
        args.document_dir,
        index_version=knowledge_base.embedding_profile.index_version,
        chunk_size=args.literature_chunk_size,
        chunk_overlap=args.literature_chunk_overlap,
    )
    literature_result = knowledge_base.ingest(literature_request)
    documents = project_result["documents"] + literature_result["documents"]
    failed = [document for document in documents if document.get("status") != "available"]
    project_search = knowledge_base.search(
        KnowledgeSearchRequest(
            query="agent retry budget repeated tool calls",
            index_version=knowledge_base.embedding_profile.index_version,
            corpus="project",
            retrieval_mode="hybrid",
            top_k=3,
        )
    )
    literature_search = knowledge_base.search(
        KnowledgeSearchRequest(
            query="YODEL maximum packing fraction interparticle forces",
            index_version=knowledge_base.embedding_profile.index_version,
            corpus="literature",
            retrieval_mode="hybrid",
            top_k=3,
        )
    )
    project_isolated = bool(project_search["hits"]) and all(
        hit.get("corpus") == "project" for hit in project_search["hits"]
    )
    literature_isolated = bool(literature_search["hits"]) and all(
        hit.get("corpus") == "literature" for hit in literature_search["hits"]
    )
    literature_citations_complete = bool(literature_search["hits"]) and all(
        hit.get("doi") and hit.get("source_url") and hit.get("page_start")
        for hit in literature_search["hits"]
    )
    verification_ok = project_isolated and literature_isolated and literature_citations_complete
    report = {
        "ok": not failed and verification_ok,
        "embedding_profile": knowledge_base.embedding_profile.model_dump(mode="json"),
        "project": {
            "documents": len(project_result["documents"]),
            "chunks": sum(int(document.get("chunk_count") or 0) for document in project_result["documents"]),
            "split_version": project_result["split_version"],
        },
        "literature": {
            "documents": len(literature_result["documents"]),
            "chunks": sum(int(document.get("chunk_count") or 0) for document in literature_result["documents"]),
            "split_version": literature_result["split_version"],
        },
        "failed": failed,
        "verification": {
            "project_corpus_isolated": project_isolated,
            "literature_corpus_isolated": literature_isolated,
            "literature_citations_complete": literature_citations_complete,
            "project_hit_sources": [Path(str(hit.get("source_path") or "")).name for hit in project_search["hits"]],
            "literature_hits": [
                {
                    "title": hit.get("title"),
                    "doi": hit.get("doi"),
                    "page_start": hit.get("page_start"),
                }
                for hit in literature_search["hits"]
            ],
        },
        "model_call_accounting": {
            "real_llm_calls": 0,
            "simulated_model_calls": 0,
            "embedding_provider": knowledge_base.embedding_profile.provider,
            "document_encode_calls": getattr(knowledge_base.embedding, "document_encode_calls", None),
            "http_encode_calls": getattr(knowledge_base.embedding, "http_encode_calls", None),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failed and verification_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
