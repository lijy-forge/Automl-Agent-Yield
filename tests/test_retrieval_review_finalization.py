from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_yieldmind_retrieval_review import finalize_reviews, write_final_outputs


FIELDS = [
    "reviewer_id",
    "row_id",
    "case_id",
    "query",
    "query_zh",
    "candidate_code",
    "candidate_text",
    "candidate_text_zh",
    "relevance",
    "confidence",
    "notes",
]


def _write_review(path: Path, reviewer_id: str, labels: list[str], confidences: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (label, confidence) in enumerate(zip(labels, confidences)):
            writer.writerow(
                {
                    "reviewer_id": reviewer_id,
                    "row_id": f"case:{chr(ord('A') + index)}",
                    "case_id": "case",
                    "query": "query",
                    "query_zh": "query zh",
                    "candidate_code": chr(ord("A") + index),
                    "candidate_text": f"candidate {index}",
                    "candidate_text_zh": f"candidate zh {index}",
                    "relevance": label,
                    "confidence": confidence,
                    "notes": f"note {reviewer_id} {index}",
                }
            )


def _write_manifest(path: Path) -> None:
    rows = []
    for index in range(3):
        text = f"candidate {index}"
        rows.append(
            {
                "row_id": f"case:{chr(ord('A') + index)}",
                "case_id": "case",
                "retrieval_rank": index + 1,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    path.write_text(json.dumps({"version": "test", "rows": rows}), encoding="utf-8")


def _write_adjudication(
    path: Path,
    *,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    manifest_path: Path,
    label: str = "1",
    change_query: bool = False,
) -> None:
    from scripts.analyze_yieldmind_retrieval_reviews import analyze_reviews

    _, disagreements = analyze_reviews(
        reviewer_a_path=reviewer_a_path,
        reviewer_b_path=reviewer_b_path,
        manifest_path=manifest_path,
        reviewer_a_name="reviewer_1",
        reviewer_b_name="reviewer_2",
    )
    disagreements[0]["adjudicated_relevance"] = label
    disagreements[0]["adjudication_notes"] = "reviewed against the rubric"
    if change_query:
        disagreements[0]["query"] = "changed query"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(disagreements[0]))
        writer.writeheader()
        writer.writerows(disagreements)


def test_finalize_reviews_writes_gold_metrics_and_bad_cases(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    manifest = tmp_path / "manifest.json"
    adjudication = tmp_path / "adjudication.csv"
    _write_review(first, "reviewer_1", ["1", "2", "0"], ["high", "medium", "low"])
    _write_review(second, "reviewer_2", ["0", "2", "0"], ["high", "medium", "low"])
    _write_manifest(manifest)
    _write_adjudication(
        adjudication,
        reviewer_a_path=first,
        reviewer_b_path=second,
        manifest_path=manifest,
    )

    report, gold_rows = finalize_reviews(
        reviewer_a_path=first,
        reviewer_b_path=second,
        manifest_path=manifest,
        adjudication_path=adjudication,
        reviewer_a_name="reviewer_1",
        reviewer_b_name="reviewer_2",
    )

    assert report["adjudication"]["complete"] is True
    assert report["adjudication"]["row_count"] == 1
    assert report["final_gold"]["metrics"]["mrr_direct"] == 0.5
    assert report["final_gold"]["top1_bad_case_count"] == 1
    assert report["bad_cases"][0]["failure_type"] == "ranking_error"
    assert [row["final_relevance"] for row in gold_rows] == [1, 2, 0]

    report_path = tmp_path / "final.json"
    gold_path = tmp_path / "gold.csv"
    write_final_outputs(report, gold_rows, report_path=report_path, gold_path=gold_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["real_llm_calls"] == 0
    assert len(list(csv.DictReader(gold_path.open(encoding="utf-8-sig")))) == 3


def test_finalize_reviews_rejects_changed_protected_field(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    manifest = tmp_path / "manifest.json"
    adjudication = tmp_path / "adjudication.csv"
    _write_review(first, "reviewer_1", ["2", "1", "0"], ["high", "medium", "low"])
    _write_review(second, "reviewer_2", ["2", "0", "0"], ["high", "medium", "low"])
    _write_manifest(manifest)
    _write_adjudication(
        adjudication,
        reviewer_a_path=first,
        reviewer_b_path=second,
        manifest_path=manifest,
        change_query=True,
    )

    with pytest.raises(ValueError, match="protected fields changed"):
        finalize_reviews(
            reviewer_a_path=first,
            reviewer_b_path=second,
            manifest_path=manifest,
            adjudication_path=adjudication,
            reviewer_a_name="reviewer_1",
            reviewer_b_name="reviewer_2",
        )


def test_finalize_reviews_rejects_blank_label(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    manifest = tmp_path / "manifest.json"
    adjudication = tmp_path / "adjudication.csv"
    _write_review(first, "reviewer_1", ["2", "1", "0"], ["high", "medium", "low"])
    _write_review(second, "reviewer_2", ["2", "0", "0"], ["high", "medium", "low"])
    _write_manifest(manifest)
    _write_adjudication(
        adjudication,
        reviewer_a_path=first,
        reviewer_b_path=second,
        manifest_path=manifest,
        label="",
    )

    with pytest.raises(ValueError, match="blank or invalid"):
        finalize_reviews(
            reviewer_a_path=first,
            reviewer_b_path=second,
            manifest_path=manifest,
            adjudication_path=adjudication,
            reviewer_a_name="reviewer_1",
            reviewer_b_name="reviewer_2",
        )
