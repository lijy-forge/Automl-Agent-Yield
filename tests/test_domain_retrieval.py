from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_yield import _compact_search_report_for_prompt, run_search_stage
from yieldmind.database import YieldMindStore
from yieldmind.domain_retrieval import (
    build_domain_knowledge_queries,
    retrieve_domain_knowledge,
)
from yieldmind.domain_workflow import RealYieldDomainAdapter, YieldDomainWorkflow
from yieldmind.knowledge_base import (
    KnowledgeBase,
    KnowledgeIngestRequest,
    KnowledgeSourceMetadata,
)


def _knowledge_fixture(tmp_path: Path) -> tuple[YieldMindStore, KnowledgeBase]:
    project = tmp_path / "project.md"
    project.write_text(
        "# Workflow\n\nLangGraph Manager routes bounded retries and validates artifacts.",
        encoding="utf-8",
    )
    literature = tmp_path / "literature.md"
    literature.write_text(
        "# Rheology\n\nYODEL relates suspension yield stress to packing density phi_m, "
        "solid volume fraction, and particle contact networks. " * 6,
        encoding="utf-8",
    )
    store = YieldMindStore(tmp_path / "yieldmind.sqlite3")
    knowledge_base = KnowledgeBase(
        store=store,
        chroma_dir=tmp_path / "chroma",
        collection_name="domain_mainline",
    )
    knowledge_base.ingest(
        KnowledgeIngestRequest(
            paths=[str(project), str(literature)],
            chunk_size=260,
            chunk_overlap=40,
            split_strategy="section_aware_v4",
            source_metadata={
                str(project): KnowledgeSourceMetadata(corpus="project", title="Workflow rules"),
                str(literature): KnowledgeSourceMetadata(
                    corpus="literature",
                    title="YODEL packing evidence",
                    doi="10.1000/yodel-test",
                    source_url="https://example.test/yodel",
                ),
            },
        )
    )
    return store, knowledge_base


def test_domain_knowledge_retrieval_routes_validates_and_budgets(tmp_path: Path) -> None:
    store, knowledge_base = _knowledge_fixture(tmp_path)
    queries = build_domain_knowledge_queries(
        "Build a yield stress model for a high-solid suspension",
        None,
        {"columns": ["phi", "sp_percent", "yield_stress"]},
    )

    report = retrieve_domain_knowledge(
        knowledge_base,
        store,
        queries=queries,
        top_k=3,
        context_budget_chars=1500,
    )

    assert report["status"] == "passed"
    assert report["retrieval_mode"] == "hybrid"
    assert report["routing_mode"] == "auto"
    assert report["context_chars_used"] <= 1500
    assert report["evidence_validation"]["ok"] is True
    assert report["evidence_validation"]["valid_count"] >= len(report["snippets"])
    assert report["selected_valid_count"] == len(report["snippets"])
    assert report["snippets"]
    first = report["snippets"][0]
    assert first["source_id"].startswith("KB:chk_")
    assert first["evidence_origin"] == "knowledge_base"
    assert first["text_hash"]
    assert first["document_version"]
    assert first["index_version"] == knowledge_base.embedding_profile.index_version
    assert first["routing"]["resolved_corpus"] == "literature"


def test_knowledge_evidence_enters_existing_agent_snippet_contract(tmp_path: Path) -> None:
    store, knowledge_base = _knowledge_fixture(tmp_path)
    knowledge_report = retrieve_domain_knowledge(
        knowledge_base,
        store,
        queries=["YODEL suspension yield stress packing density phi_m"],
        top_k=2,
        context_budget_chars=1200,
    )

    search_report = run_search_stage(
        "Build a yield stress model",
        {"schema": {"source_schema": "test"}, "columns": ["phi"]},
        False,
        None,
        knowledge_report,
    )
    compact = _compact_search_report_for_prompt(search_report)

    assert search_report["evidence_counts"] == {
        "knowledge": len(knowledge_report["snippets"]),
        "external": 0,
        "combined": len(knowledge_report["snippets"]),
    }
    assert compact["snippets"][0]["evidence_origin"] == "knowledge_base"
    assert compact["snippets"][0]["chunk_id"]
    assert compact["snippets"][0]["text_hash"]
    assert compact["knowledge_search"]["evidence_validation"]["ok"] is True


def test_real_domain_adapter_calls_knowledge_mainline(tmp_path: Path, monkeypatch) -> None:
    store, knowledge_base = _knowledge_fixture(tmp_path)
    captured: dict[str, Any] = {}

    class FakeManager:
        args = SimpleNamespace(
            prompt="Build a YODEL yield stress model",
            query=None,
        )
        train_profile = {"columns": ["phi", "sp_percent", "yield_stress"]}

        def _run_search_agent(self, knowledge_search_report=None) -> bool:
            captured["report"] = knowledge_search_report
            return True

    monkeypatch.setattr("yieldmind.domain_workflow._hydrate_manager", lambda *args, **kwargs: FakeManager())
    adapter = RealYieldDomainAdapter(
        store=store,
        knowledge_base_factory=lambda: knowledge_base,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = adapter.search(
        {
            "run_dir": str(run_dir),
            "use_knowledge_search": True,
            "knowledge_top_k": 3,
            "knowledge_context_budget_chars": 1500,
        }
    )

    assert result["ok"] is True
    assert captured["report"]["status"] == "passed"
    assert captured["report"]["snippets"]
    assert result["knowledge_search_report"] == captured["report"]
    summary = YieldDomainWorkflow._result_summary("search", result)
    assert summary["knowledge_search"]["valid_evidence_refs"] == len(captured["report"]["snippets"])
    assert summary["knowledge_search"]["query_routes"][0]["resolved_corpus"] == "literature"
    assert summary["knowledge_search"]["top_evidence"][0]["source_id"].startswith("KB:")
