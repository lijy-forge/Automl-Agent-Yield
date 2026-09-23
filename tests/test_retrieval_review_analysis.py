from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_yieldmind_retrieval_reviews import analyze_reviews, write_analysis


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


def test_analyze_reviews_validates_and_reports_disagreement(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    manifest = tmp_path / "manifest.json"
    _write_review(first, "reviewer_1", ["2", "1", "0"], ["high", "middle", "low"])
    _write_review(second, "reviewer_1", ["2", "0", "0"], ["high", "medium", "low"])
    _write_manifest(manifest)

    report, disagreements = analyze_reviews(
        reviewer_a_path=first,
        reviewer_b_path=second,
        manifest_path=manifest,
        reviewer_a_name="reviewer_1",
        reviewer_b_name="reviewer_2",
    )

    assert report["agreement"]["exact_count"] == 2
    assert report["agreement"]["disagreement_count"] == 1
    assert report["agreement"]["severe_0_vs_2_count"] == 0
    assert report["agreement"]["normalized_confidence_agreement"] == 1.0
    assert report["reviewers"]["reviewer_2"]["source_reviewer_ids"] == {"reviewer_1": 3}
    assert report["conservative_consensus"]["metrics"]["top1_direct"] == 1.0
    assert disagreements[0]["row_id"] == "case:B"
    assert disagreements[0]["adjudicated_relevance"] == ""

    report_path = tmp_path / "report.json"
    adjudication_path = tmp_path / "adjudication.csv"
    write_analysis(report, disagreements, report_path=report_path, adjudication_path=adjudication_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["real_llm_calls"] == 0
    assert len(list(csv.DictReader(adjudication_path.open(encoding="utf-8-sig", newline="")))) == 1


def test_analyze_reviews_rejects_changed_candidate_text(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    manifest = tmp_path / "manifest.json"
    _write_review(first, "reviewer_1", ["2", "1", "0"], ["high", "medium", "low"])
    _write_review(second, "reviewer_2", ["2", "1", "0"], ["high", "medium", "low"])
    _write_manifest(manifest)
    content = second.read_text(encoding="utf-8-sig").replace("candidate 1", "changed candidate")
    second.write_text(content, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="hashes differ"):
        analyze_reviews(
            reviewer_a_path=first,
            reviewer_b_path=second,
            manifest_path=manifest,
            reviewer_a_name="reviewer_1",
            reviewer_b_name="reviewer_2",
        )
