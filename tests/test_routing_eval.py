from __future__ import annotations

import json
from pathlib import Path

from scripts.run_yieldmind_routing_eval import evaluate_routing_cases
from yieldmind.knowledge_base import KnowledgeBase, KnowledgeSearchRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_routing_intent_benchmark_passes_all_fixed_cases() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "evals" / "yieldmind_routing_cases_v1.json").read_text(encoding="utf-8")
    )
    report = evaluate_routing_cases(payload)

    assert report["status"] == "passed"
    assert report["case_count"] == 32
    assert report["accuracy"] == 1.0
    assert report["forced_route_rate_on_ambiguous"] == 0.0
    assert report["real_llm_calls"] == 0


def test_routing_terms_use_boundaries_and_explicit_corpus_still_wins() -> None:
    _, rapid = KnowledgeBase._route_request(
        KnowledgeSearchRequest(query="rapid optimization notes", routing_mode="auto")
    )
    _, redistribution = KnowledgeBase._route_request(
        KnowledgeSearchRequest(
            query="redistribution of suspension particles during shear",
            routing_mode="auto",
        )
    )
    _, explicit = KnowledgeBase._route_request(
        KnowledgeSearchRequest(
            query="Docker workflow",
            corpus="literature",
            routing_mode="auto",
        )
    )

    assert rapid["resolved_corpus"] is None
    assert redistribution["resolved_corpus"] == "literature"
    assert explicit["resolved_corpus"] == "literature"
    assert explicit["reason"] == "explicit_corpus_filter"
