#!/usr/bin/env python3
"""Evaluate deterministic project/literature corpus routing without model calls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import json_dumps
from yieldmind.database import YieldMindStore
from yieldmind.knowledge_base import KnowledgeBase, KnowledgeSearchRequest
from yieldmind.knowledge_runtime import configured_knowledge_base


DEFAULT_CASES = PROJECT_ROOT / "evals" / "yieldmind_routing_cases_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "routing_evals"


def evaluate_routing_cases(payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 3:
        raise ValueError("Routing benchmark requires at least three cases.")
    ids = [str(case.get("id") or "") for case in cases]
    if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Routing case IDs must be non-empty and unique.")

    diagnostics: list[dict[str, Any]] = []
    confusion: Counter[str] = Counter()
    class_totals: Counter[str] = Counter()
    class_passed: Counter[str] = Counter()
    for case in cases:
        expected = case.get("expected_corpus")
        if expected not in {None, "project", "literature"}:
            raise ValueError(f"Invalid expected_corpus for {case['id']!r}: {expected!r}.")
        query = str(case.get("query") or "").strip()
        if not query:
            raise ValueError(f"Routing case {case['id']!r} has an empty query.")
        _, routing = KnowledgeBase._route_request(
            KnowledgeSearchRequest(query=query, routing_mode="auto")
        )
        actual = routing["resolved_corpus"]
        expected_label = expected or "all"
        actual_label = actual or "all"
        passed = actual == expected
        confusion[f"{expected_label}->{actual_label}"] += 1
        class_totals[expected_label] += 1
        class_passed[expected_label] += int(passed)
        diagnostics.append(
            {
                "id": case["id"],
                "query": query,
                "expected_corpus": expected,
                "resolved_corpus": actual,
                "passed": passed,
                "reason": routing["reason"],
                "project_score": routing["project_score"],
                "literature_score": routing["literature_score"],
            }
        )

    passed_count = sum(item["passed"] for item in diagnostics)
    return {
        "status": "passed" if passed_count == len(diagnostics) else "failed",
        "benchmark_version": payload.get("version", ""),
        "label_scope": payload.get("label_scope", ""),
        "case_count": len(diagnostics),
        "passed_count": passed_count,
        "accuracy": passed_count / len(diagnostics),
        "accuracy_by_expected_corpus": {
            label: class_passed[label] / total for label, total in sorted(class_totals.items())
        },
        "confusion": dict(sorted(confusion.items())),
        "forced_route_rate_on_ambiguous": (
            1.0 - class_passed["all"] / class_totals["all"] if class_totals["all"] else 0.0
        ),
        "cases": diagnostics,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
    }


def verify_routed_retrieval(payload: dict[str, Any]) -> dict[str, Any]:
    """Exercise the configured persistent index after the intent-only check."""
    knowledge_base = configured_knowledge_base(YieldMindStore())
    diagnostics: list[dict[str, Any]] = []
    for case in payload["cases"]:
        expected = case.get("expected_corpus")
        result = knowledge_base.search(
            KnowledgeSearchRequest(
                query=str(case["query"]),
                index_version=knowledge_base.embedding_profile.index_version,
                retrieval_mode="hybrid",
                routing_mode="auto",
                top_k=3,
                expand_parent=True,
                deduplicate_parents=True,
                context_budget_chars=12000,
            )
        )
        hits = result["hits"]
        resolved = result["routing"]["resolved_corpus"]
        corpus_pure = expected is None or all(hit.get("corpus") == expected for hit in hits)
        passed = resolved == expected and bool(hits) and corpus_pure
        diagnostics.append(
            {
                "id": case["id"],
                "expected_corpus": expected,
                "resolved_corpus": resolved,
                "hit_count": len(hits),
                "hit_corpora": sorted({str(hit.get("corpus") or "") for hit in hits}),
                "corpus_pure": corpus_pure,
                "fallback_to_all_corpora": result["routing"]["fallback_to_all_corpora"],
                "passed": passed,
            }
        )
    passed_count = sum(item["passed"] for item in diagnostics)
    routed_cases = [item for item in diagnostics if item["expected_corpus"] is not None]
    return {
        "status": "passed" if passed_count == len(diagnostics) else "failed",
        "case_count": len(diagnostics),
        "passed_count": passed_count,
        "corpus_purity_rate": (
            sum(item["corpus_pure"] for item in routed_cases) / len(routed_cases)
            if routed_cases
            else 1.0
        ),
        "corpus_purity_case_count": len(routed_cases),
        "cases": diagnostics,
        "embedding_profile": knowledge_base.embedding_profile.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--verify-retrieval", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    report = evaluate_routing_cases(payload)
    if args.verify_retrieval:
        report["retrieval_verification"] = verify_routed_retrieval(payload)
        if report["retrieval_verification"]["status"] != "passed":
            report["status"] = "failed"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"routing_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json_dumps(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": str(output_path),
                "accuracy": report["accuracy"],
                "accuracy_by_expected_corpus": report["accuracy_by_expected_corpus"],
                "failed_case_ids": [case["id"] for case in report["cases"] if not case["passed"]],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
