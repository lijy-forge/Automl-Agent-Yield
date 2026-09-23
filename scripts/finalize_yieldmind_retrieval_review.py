#!/usr/bin/env python3
"""Validate adjudication and produce final retrieval gold labels and metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.analyze_yieldmind_retrieval_reviews import (
        _read_review,
        _compare_system_top1,
        _manifest_rows_for_system,
        _retrieval_metrics_by_system,
        _single_system_metrics,
        _sha256_file,
        analyze_reviews,
    )
else:
    from analyze_yieldmind_retrieval_reviews import (  # type: ignore[no-redef]
        _read_review,
        _compare_system_top1,
        _manifest_rows_for_system,
        _retrieval_metrics_by_system,
        _single_system_metrics,
        _sha256_file,
        analyze_reviews,
    )


FINAL_FIELDS = (
    "row_id",
    "case_id",
    "query",
    "query_zh",
    "candidate_code",
    "candidate_text",
    "candidate_text_zh",
    "chunk_id",
    "source_path",
    "chunk_index",
    "retrieval_rank",
    "pool_display_rank",
    "system_memberships",
    "reviewer_a_relevance",
    "reviewer_b_relevance",
    "final_relevance",
    "resolution",
    "adjudication_notes",
)


def _read_adjudication(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Adjudication file is empty: {path}")
    required = {"row_id", "adjudicated_relevance", "adjudication_notes"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Adjudication file is missing columns: {sorted(missing)}")
    return rows


def _validate_adjudication(
    rows: list[dict[str, str]],
    expected_disagreements: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    row_ids = [row["row_id"].strip() for row in rows]
    duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Adjudication has duplicate row IDs: {duplicates[:10]}")

    expected_by_id = {str(row["row_id"]): row for row in expected_disagreements}
    actual_by_id = {row["row_id"].strip(): row for row in rows}
    if set(actual_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(actual_by_id))
        extra = sorted(set(actual_by_id) - set(expected_by_id))
        raise ValueError(f"Adjudication row IDs differ from disagreements; missing={missing[:10]}, extra={extra[:10]}")

    protected_fields = tuple(
        field
        for field in expected_disagreements[0]
        if field not in {"adjudicated_relevance", "adjudication_notes"}
    )
    changed: list[dict[str, Any]] = []
    invalid_labels: list[str] = []
    blank_notes: list[str] = []
    for row_id, expected in expected_by_id.items():
        actual = actual_by_id[row_id]
        changed_fields = [
            field for field in protected_fields if str(actual.get(field, "")) != str(expected[field])
        ]
        if changed_fields:
            changed.append({"row_id": row_id, "fields": changed_fields})
        label = actual["adjudicated_relevance"].strip()
        if label not in {"0", "1", "2"}:
            invalid_labels.append(row_id)
        if not actual["adjudication_notes"].strip():
            blank_notes.append(row_id)
    if changed:
        raise ValueError(f"Adjudication protected fields changed: {changed[:10]}")
    if invalid_labels:
        raise ValueError(f"Adjudication has blank or invalid relevance labels: {invalid_labels[:10]}")
    if blank_notes:
        raise ValueError(f"Adjudication notes are required: {blank_notes[:10]}")
    return actual_by_id


def _build_bad_cases(
    final_labels: dict[str, int],
    manifest_rows: dict[str, dict[str, Any]],
    review_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_id, metadata in manifest_rows.items():
        grouped[str(metadata["case_id"])].append(row_id)

    bad_cases: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        ranked = sorted(grouped[case_id], key=lambda row_id: int(manifest_rows[row_id]["retrieval_rank"]))
        top_id = ranked[0]
        if final_labels[top_id] >= 2:
            continue
        direct_ids = [row_id for row_id in ranked if final_labels[row_id] >= 2]
        first_direct_id = direct_ids[0] if direct_ids else None
        top_metadata = manifest_rows[top_id]
        direct_metadata = manifest_rows[first_direct_id] if first_direct_id else None
        bad_cases.append(
            {
                "case_id": case_id,
                "query": review_rows[top_id]["query"],
                "query_zh": review_rows[top_id]["query_zh"],
                "top1_row_id": top_id,
                "top1_relevance": final_labels[top_id],
                "top1_chunk_id": top_metadata.get("chunk_id"),
                "top1_source_path": top_metadata.get("source_path"),
                "first_direct_rank": int(direct_metadata["retrieval_rank"]) if direct_metadata else None,
                "first_direct_row_id": first_direct_id,
                "first_direct_chunk_id": direct_metadata.get("chunk_id") if direct_metadata else None,
                "first_direct_source_path": direct_metadata.get("source_path") if direct_metadata else None,
                "failure_type": "ranking_error" if first_direct_id else "no_direct_candidate_in_top5",
            }
        )
    return bad_cases


def finalize_reviews(
    *,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    manifest_path: Path,
    adjudication_path: Path,
    reviewer_a_name: str,
    reviewer_b_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dual_report, expected_disagreements = analyze_reviews(
        reviewer_a_path=reviewer_a_path,
        reviewer_b_path=reviewer_b_path,
        manifest_path=manifest_path,
        reviewer_a_name=reviewer_a_name,
        reviewer_b_name=reviewer_b_name,
    )
    adjudication_by_id = _validate_adjudication(
        _read_adjudication(adjudication_path), expected_disagreements
    )
    reviewer_a = _read_review(reviewer_a_path)
    reviewer_b = _read_review(reviewer_b_path)
    rows_a = {row["row_id"]: row for row in reviewer_a}
    rows_b = {row["row_id"]: row for row in reviewer_b}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = {str(row["row_id"]): row for row in manifest["rows"]}
    pooled_system_ids = sorted(
        {
            system_id
            for metadata in manifest_rows.values()
            for system_id in (metadata.get("systems") or {})
        }
    )

    final_labels: dict[str, int] = {}
    gold_rows: list[dict[str, Any]] = []
    resolved_to_a = 0
    resolved_to_b = 0
    for row_id, row_a in rows_a.items():
        label_a = int(row_a["relevance"])
        label_b = int(rows_b[row_id]["relevance"])
        adjudication = adjudication_by_id.get(row_id)
        if label_a == label_b:
            final_label = label_a
            resolution = "agreement"
            adjudication_notes = ""
        else:
            if adjudication is None:
                raise ValueError(f"Missing adjudication for disagreement: {row_id}")
            final_label = int(adjudication["adjudicated_relevance"])
            resolution = "adjudication"
            adjudication_notes = adjudication["adjudication_notes"].strip()
            resolved_to_a += final_label == label_a
            resolved_to_b += final_label == label_b
        final_labels[row_id] = final_label
        metadata = manifest_rows[row_id]
        gold_rows.append(
            {
                "row_id": row_id,
                "case_id": row_a["case_id"],
                "query": row_a["query"],
                "query_zh": row_a["query_zh"],
                "candidate_code": row_a["candidate_code"],
                "candidate_text": row_a["candidate_text"],
                "candidate_text_zh": row_a["candidate_text_zh"],
                "chunk_id": metadata.get("chunk_id", ""),
                "source_path": metadata.get("source_path", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "retrieval_rank": "" if pooled_system_ids else int(metadata["retrieval_rank"]),
                "pool_display_rank": metadata.get("pool_display_rank", ""),
                "system_memberships": (
                    json.dumps(metadata.get("systems", {}), ensure_ascii=False, sort_keys=True)
                    if pooled_system_ids
                    else ""
                ),
                "reviewer_a_relevance": label_a,
                "reviewer_b_relevance": label_b,
                "final_relevance": final_label,
                "resolution": resolution,
                "adjudication_notes": adjudication_notes,
            }
        )

    gold_rows.sort(
        key=lambda row: (
            row["case_id"],
            int(row["pool_display_rank"] or row["retrieval_rank"]),
            row["row_id"],
        )
    )
    bad_cases = [] if pooled_system_ids else _build_bad_cases(final_labels, manifest_rows, rows_a)
    bad_cases_by_system = {
        system_id: _build_bad_cases(
            final_labels,
            _manifest_rows_for_system(manifest_rows, system_id),
            rows_a,
        )
        for system_id in pooled_system_ids
    }
    report = {
        "version": "yieldmind-retrieval-final-gold-v1",
        "inputs": {
            "reviewer_a": {"name": reviewer_a_name, "file": reviewer_a_path.name, "sha256": _sha256_file(reviewer_a_path)},
            "reviewer_b": {"name": reviewer_b_name, "file": reviewer_b_path.name, "sha256": _sha256_file(reviewer_b_path)},
            "manifest": {"file": manifest_path.name, "sha256": _sha256_file(manifest_path)},
            "adjudication": {"file": adjudication_path.name, "sha256": _sha256_file(adjudication_path)},
        },
        "agreement": dual_report["agreement"],
        "adjudication": {
            "row_count": len(adjudication_by_id),
            "label_distribution": dict(
                sorted(Counter(row["adjudicated_relevance"].strip() for row in adjudication_by_id.values()).items())
            ),
            "resolved_to_reviewer_a": resolved_to_a,
            "resolved_to_reviewer_b": resolved_to_b,
            "complete": True,
        },
        "final_gold": {
            "row_count": len(gold_rows),
            "case_count": len({row["case_id"] for row in gold_rows}),
            "relevance_distribution": dict(sorted(Counter(final_labels.values()).items())),
            "metrics": _single_system_metrics(final_labels, manifest_rows),
            "system_metrics": _retrieval_metrics_by_system(final_labels, manifest_rows),
            "system_top1_comparison": _compare_system_top1(final_labels, manifest_rows),
            "top1_bad_case_count": len(bad_cases) if not pooled_system_ids else None,
            "top1_bad_case_count_by_system": {
                system_id: len(items) for system_id, items in bad_cases_by_system.items()
            },
        },
        "bad_cases": bad_cases,
        "bad_cases_by_system": bad_cases_by_system,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
    }
    return report, gold_rows


def write_final_outputs(
    report: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    *,
    report_path: Path,
    gold_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(gold_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adjudication", required=True)
    parser.add_argument("--reviewer-a-name", default="reviewer_1")
    parser.add_argument("--reviewer-b-name", default="reviewer_2")
    parser.add_argument("--report", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()
    report, gold_rows = finalize_reviews(
        reviewer_a_path=Path(args.reviewer_a),
        reviewer_b_path=Path(args.reviewer_b),
        manifest_path=Path(args.manifest),
        adjudication_path=Path(args.adjudication),
        reviewer_a_name=args.reviewer_a_name,
        reviewer_b_name=args.reviewer_b_name,
    )
    write_final_outputs(report, gold_rows, report_path=Path(args.report), gold_path=Path(args.gold))
    print(
        json.dumps(
            {
                "ok": True,
                "metrics": report["final_gold"]["metrics"],
                "bad_case_count": report["final_gold"]["top1_bad_case_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
