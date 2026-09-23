from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.run_yieldmind_reranker_eval import evaluate_reranker_scores, load_gold_rows, run


FIELDS = [
    "row_id",
    "case_id",
    "query",
    "candidate_text",
    "chunk_id",
    "source_path",
    "retrieval_rank",
    "final_relevance",
]


def _write_gold(path: Path, *, inconsistent_query: bool = False) -> None:
    labels = {"case_a": [0, 2, 1], "case_b": [2, 0, 0]}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for case_id, relevance in labels.items():
            for index, label in enumerate(relevance, start=1):
                query = f"query {case_id}"
                if inconsistent_query and case_id == "case_a" and index == 3:
                    query = "changed query"
                writer.writerow(
                    {
                        "row_id": f"{case_id}:{index}",
                        "case_id": case_id,
                        "query": query,
                        "candidate_text": f"document {case_id} {index}",
                        "chunk_id": f"chunk-{case_id}-{index}",
                        "source_path": f"{case_id}.md",
                        "retrieval_rank": index,
                        "final_relevance": label,
                    }
                )


def test_reranker_gold_replay_improves_top1_without_changing_recall(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    _write_gold(gold)
    rows = load_gold_rows(gold)
    scores = [0.1, 0.9, 0.2, 0.9, 0.2, 0.1]

    comparison, cases = evaluate_reranker_scores(rows, scores)

    assert comparison["original_metrics"]["recall_at_5_direct"] == 1.0
    assert comparison["reranked_metrics"]["recall_at_5_direct"] == 1.0
    assert comparison["original_metrics"]["top1_direct"] == 0.5
    assert comparison["reranked_metrics"]["top1_direct"] == 1.0
    assert comparison["metric_deltas"]["mrr_direct"] == 0.25
    assert comparison["direct_relevance_promoted_to_top1"] == 1
    assert comparison["direct_relevance_demoted_from_top1"] == 0
    assert cases[0]["reranked_top1_row_id"] == "case_a:2"


def test_reranker_ties_preserve_original_rank_and_report_input_hash(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    _write_gold(gold)

    def tied_scorer(pairs):
        return [0.5] * len(pairs), {"pair_count": len(pairs), "mean_pair_latency_ms": 0.0}

    report = run(
        gold_path=gold,
        scorer=tied_scorer,
        model={"scorer_kind": "cross_encoder", "model_id": "test", "revision": "immutable-test-revision"},
    )

    assert report["comparison"]["top1_changed_count"] == 0
    assert report["comparison"]["metric_deltas"]["ndcg_at_5"] == 0.0
    assert len(report["gold_input"]["sha256"]) == 64
    assert report["real_reranker_model_calls"] == 6
    assert report["real_embedding_scoring_pairs"] == 0
    assert report["decision"]["default_retrieval_changed"] is False


def test_reranker_gold_validation_rejects_inconsistent_queries_and_nonfinite_scores(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    _write_gold(gold, inconsistent_query=True)
    with pytest.raises(ValueError, match="inconsistent queries"):
        load_gold_rows(gold)

    _write_gold(gold)
    rows = load_gold_rows(gold)
    with pytest.raises(ValueError, match="finite"):
        evaluate_reranker_scores(rows, [float("nan")] * len(rows))
