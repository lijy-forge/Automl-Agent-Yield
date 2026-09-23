#!/usr/bin/env python3
"""Validate two blind-review files and produce agreement/adjudication artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score, confusion_matrix


IMMUTABLE_FIELDS = (
    "row_id",
    "case_id",
    "query",
    "query_zh",
    "candidate_code",
    "candidate_text",
    "candidate_text_zh",
)
VALID_RELEVANCE = {"0", "1", "2"}
CONFIDENCE_ALIASES = {"low": "low", "middle": "medium", "medium": "medium", "high": "high"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Review file is empty: {path}")
    missing = set(IMMUTABLE_FIELDS + ("reviewer_id", "relevance", "confidence", "notes")) - set(rows[0])
    if missing:
        raise ValueError(f"Review file is missing columns {sorted(missing)}: {path}")
    return rows


def _validate_review(
    rows: list[dict[str, str]],
    manifest_rows: dict[str, dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    row_ids = [row["row_id"].strip() for row in rows]
    duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} has duplicate row IDs: {duplicates[:10]}")
    if set(row_ids) != set(manifest_rows):
        missing = sorted(set(manifest_rows) - set(row_ids))
        extra = sorted(set(row_ids) - set(manifest_rows))
        raise ValueError(f"{label} row IDs differ from manifest; missing={missing[:10]}, extra={extra[:10]}")

    invalid_relevance = sorted({row["relevance"].strip() for row in rows} - VALID_RELEVANCE)
    invalid_confidence = sorted({row["confidence"].strip().lower() for row in rows} - set(CONFIDENCE_ALIASES))
    if invalid_relevance or invalid_confidence:
        raise ValueError(
            f"{label} has invalid values; relevance={invalid_relevance}, confidence={invalid_confidence}"
        )

    text_hash_mismatches = []
    for row in rows:
        expected = str(manifest_rows[row["row_id"]].get("text_sha256") or "")
        actual = hashlib.sha256(row["candidate_text"].encode("utf-8")).hexdigest()
        if actual != expected:
            text_hash_mismatches.append(row["row_id"])
    if text_hash_mismatches:
        raise ValueError(f"{label} candidate text hashes differ from manifest: {text_hash_mismatches[:10]}")

    confidence_values = [row["confidence"].strip().lower() for row in rows]
    return {
        "row_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "source_reviewer_ids": dict(Counter(row["reviewer_id"].strip() for row in rows)),
        "relevance_distribution": dict(Counter(row["relevance"].strip() for row in rows)),
        "confidence_distribution_raw": dict(Counter(confidence_values)),
        "confidence_alias_normalizations": sum(value == "middle" for value in confidence_values),
        "notes_filled": sum(bool(row["notes"].strip()) for row in rows),
    }


def _retrieval_metrics(
    relevance_by_row: dict[str, int],
    manifest_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row_id, metadata in manifest_rows.items():
        grouped[str(metadata["case_id"])].append(row_id)

    reciprocal_direct: list[float] = []
    reciprocal_partial: list[float] = []
    ndcg: list[float] = []
    top_labels: list[int] = []
    cases_without_direct: list[str] = []
    for case_id, row_ids in grouped.items():
        ranked = sorted(row_ids, key=lambda row_id: int(manifest_rows[row_id]["retrieval_rank"]))
        labels = [int(relevance_by_row[row_id]) for row_id in ranked]
        top_labels.append(labels[0])
        direct_ranks = [rank for rank, value in enumerate(labels, start=1) if value >= 2]
        partial_ranks = [rank for rank, value in enumerate(labels, start=1) if value >= 1]
        reciprocal_direct.append(1.0 / direct_ranks[0] if direct_ranks else 0.0)
        reciprocal_partial.append(1.0 / partial_ranks[0] if partial_ranks else 0.0)
        if not direct_ranks:
            cases_without_direct.append(case_id)

        dcg = sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(labels, start=1))
        ideal = sorted(labels, reverse=True)
        ideal_dcg = sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(ideal, start=1))
        ndcg.append(dcg / ideal_dcg if ideal_dcg else 0.0)

    case_count = len(grouped)
    return {
        "case_count": case_count,
        "recall_at_5_direct": (case_count - len(cases_without_direct)) / case_count,
        "mrr_direct": sum(reciprocal_direct) / case_count,
        "mrr_partial": sum(reciprocal_partial) / case_count,
        "top1_direct": sum(value >= 2 for value in top_labels) / case_count,
        "top1_partial": sum(value >= 1 for value in top_labels) / case_count,
        "ndcg_at_5": sum(ndcg) / case_count,
        "cases_without_direct": sorted(cases_without_direct),
    }


def _manifest_rows_for_system(
    manifest_rows: dict[str, dict[str, Any]], system_id: str
) -> dict[str, dict[str, Any]]:
    system_rows: dict[str, dict[str, Any]] = {}
    for row_id, metadata in manifest_rows.items():
        membership = (metadata.get("systems") or {}).get(system_id)
        if membership is None:
            continue
        system_rows[row_id] = {
            "case_id": metadata["case_id"],
            "retrieval_rank": int(membership["retrieval_rank"]),
            **membership,
        }
    return system_rows


def _retrieval_metrics_by_system(
    relevance_by_row: dict[str, int],
    manifest_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Score every hidden retrieval system against one shared pool of labels."""
    system_ids = sorted(
        {
            system_id
            for metadata in manifest_rows.values()
            for system_id in (metadata.get("systems") or {})
        }
    )
    return {
        system_id: _retrieval_metrics(
            relevance_by_row, _manifest_rows_for_system(manifest_rows, system_id)
        )
        for system_id in system_ids
    }


def _single_system_metrics(
    relevance_by_row: dict[str, int], manifest_rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Avoid treating randomized pool display order as a retrieval ranking."""
    if any(metadata.get("systems") for metadata in manifest_rows.values()):
        return {}
    return _retrieval_metrics(relevance_by_row, manifest_rows)


def _compare_system_top1(
    relevance_by_row: dict[str, int],
    manifest_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    system_ids = sorted(
        {
            system_id
            for metadata in manifest_rows.values()
            for system_id in (metadata.get("systems") or {})
        }
    )
    if len(system_ids) != 2:
        return {}
    top1_rows: dict[str, dict[str, str]] = {system_id: {} for system_id in system_ids}
    for row_id, metadata in manifest_rows.items():
        case_id = str(metadata["case_id"])
        for system_id, membership in (metadata.get("systems") or {}).items():
            if int(membership["retrieval_rank"]) != 1:
                continue
            if case_id in top1_rows[system_id]:
                raise ValueError(f"System {system_id!r} has multiple Top-1 rows for case {case_id!r}.")
            top1_rows[system_id][case_id] = row_id
    case_ids = sorted(set(top1_rows[system_ids[0]]) | set(top1_rows[system_ids[1]]))
    if any(set(rows) != set(case_ids) for rows in top1_rows.values()):
        raise ValueError("Pooled systems do not cover the same case IDs at Top-1.")

    first, second = system_ids
    first_wins: list[str] = []
    second_wins: list[str] = []
    ties: list[str] = []
    same_candidate: list[str] = []
    for case_id in case_ids:
        first_row = top1_rows[first][case_id]
        second_row = top1_rows[second][case_id]
        if first_row == second_row:
            same_candidate.append(case_id)
        first_label = int(relevance_by_row[first_row])
        second_label = int(relevance_by_row[second_row])
        if first_label > second_label:
            first_wins.append(case_id)
        elif second_label > first_label:
            second_wins.append(case_id)
        else:
            ties.append(case_id)
    metrics = _retrieval_metrics_by_system(relevance_by_row, manifest_rows)
    return {
        "systems": system_ids,
        "case_count": len(case_ids),
        "same_candidate_count": len(same_candidate),
        "different_candidate_count": len(case_ids) - len(same_candidate),
        "wins": {first: len(first_wins), second: len(second_wins)},
        "ties": len(ties),
        "win_case_ids": {first: first_wins, second: second_wins},
        "tie_case_ids": ties,
        "top1_direct": {
            system_id: metrics[system_id]["top1_direct"] for system_id in system_ids
        },
        "top1_direct_delta_second_minus_first": (
            metrics[second]["top1_direct"] - metrics[first]["top1_direct"]
        ),
    }


def analyze_reviews(
    *,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    manifest_path: Path,
    reviewer_a_name: str,
    reviewer_b_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reviewer_a = _read_review(reviewer_a_path)
    reviewer_b = _read_review(reviewer_b_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = {str(row["row_id"]): row for row in manifest["rows"]}
    validation_a = _validate_review(reviewer_a, manifest_rows, label=reviewer_a_name)
    validation_b = _validate_review(reviewer_b, manifest_rows, label=reviewer_b_name)
    rows_a = {row["row_id"]: row for row in reviewer_a}
    rows_b = {row["row_id"]: row for row in reviewer_b}

    immutable_changes = []
    for row_id, row_a in rows_a.items():
        changed_fields = [field for field in IMMUTABLE_FIELDS if row_a[field] != rows_b[row_id][field]]
        if changed_fields:
            immutable_changes.append({"row_id": row_id, "fields": changed_fields})
    if immutable_changes:
        raise ValueError(f"Immutable review fields differ: {immutable_changes[:10]}")

    ordered_ids = [row["row_id"] for row in reviewer_a]
    labels_a = [int(rows_a[row_id]["relevance"]) for row_id in ordered_ids]
    labels_b = [int(rows_b[row_id]["relevance"]) for row_id in ordered_ids]
    disagreements: list[dict[str, Any]] = []
    for row_id, label_a, label_b in zip(ordered_ids, labels_a, labels_b):
        if label_a == label_b:
            continue
        row_a = rows_a[row_id]
        row_b = rows_b[row_id]
        disagreements.append(
            {
                "row_id": row_id,
                "case_id": row_a["case_id"],
                "query": row_a["query"],
                "query_zh": row_a["query_zh"],
                "candidate_code": row_a["candidate_code"],
                "candidate_text": row_a["candidate_text"],
                "candidate_text_zh": row_a["candidate_text_zh"],
                "reviewer_a_relevance": label_a,
                "reviewer_a_confidence": CONFIDENCE_ALIASES[row_a["confidence"].strip().lower()],
                "reviewer_a_notes": row_a["notes"],
                "reviewer_b_relevance": label_b,
                "reviewer_b_confidence": CONFIDENCE_ALIASES[row_b["confidence"].strip().lower()],
                "reviewer_b_notes": row_b["notes"],
                "label_distance": abs(label_a - label_b),
                "adjudicated_relevance": "",
                "adjudication_notes": "",
            }
        )

    confusion = confusion_matrix(labels_a, labels_b, labels=[0, 1, 2]).tolist()
    conservative = {
        row_id: min(int(rows_a[row_id]["relevance"]), int(rows_b[row_id]["relevance"]))
        for row_id in ordered_ids
    }
    normalized_confidence_agreement = sum(
        CONFIDENCE_ALIASES[rows_a[row_id]["confidence"].strip().lower()]
        == CONFIDENCE_ALIASES[rows_b[row_id]["confidence"].strip().lower()]
        for row_id in ordered_ids
    ) / len(ordered_ids)
    report = {
        "version": "yieldmind-retrieval-dual-review-v1",
        "manifest": {
            "version": manifest.get("version"),
            "retrieval_report_sha256": manifest.get("retrieval_report_sha256"),
            "retrieval_mode": manifest.get("retrieval_mode"),
            "row_count": len(manifest_rows),
        },
        "reviewers": {
            reviewer_a_name: {
                **validation_a,
                "source_file": reviewer_a_path.name,
                "source_sha256": _sha256_file(reviewer_a_path),
                "metrics": _single_system_metrics(
                    {row_id: int(rows_a[row_id]["relevance"]) for row_id in ordered_ids}, manifest_rows
                ),
                "system_metrics": _retrieval_metrics_by_system(
                    {row_id: int(rows_a[row_id]["relevance"]) for row_id in ordered_ids}, manifest_rows
                ),
                "system_top1_comparison": _compare_system_top1(
                    {row_id: int(rows_a[row_id]["relevance"]) for row_id in ordered_ids}, manifest_rows
                ),
            },
            reviewer_b_name: {
                **validation_b,
                "source_file": reviewer_b_path.name,
                "source_sha256": _sha256_file(reviewer_b_path),
                "metrics": _single_system_metrics(
                    {row_id: int(rows_b[row_id]["relevance"]) for row_id in ordered_ids}, manifest_rows
                ),
                "system_metrics": _retrieval_metrics_by_system(
                    {row_id: int(rows_b[row_id]["relevance"]) for row_id in ordered_ids}, manifest_rows
                ),
                "system_top1_comparison": _compare_system_top1(
                    {row_id: int(rows_b[row_id]["relevance"]) for row_id in ordered_ids}, manifest_rows
                ),
            },
        },
        "agreement": {
            "exact_count": len(ordered_ids) - len(disagreements),
            "exact_rate": (len(ordered_ids) - len(disagreements)) / len(ordered_ids),
            "disagreement_count": len(disagreements),
            "severe_0_vs_2_count": sum(item["label_distance"] == 2 for item in disagreements),
            "linear_weighted_cohen_kappa": float(cohen_kappa_score(labels_a, labels_b, weights="linear")),
            "quadratic_weighted_cohen_kappa": float(
                cohen_kappa_score(labels_a, labels_b, weights="quadratic")
            ),
            "confusion_matrix": {"labels": [0, 1, 2], "rows": reviewer_a_name, "columns": reviewer_b_name, "values": confusion},
            "normalized_confidence_agreement": normalized_confidence_agreement,
            "notes_exact_match_count": sum(
                rows_a[row_id]["notes"] == rows_b[row_id]["notes"] for row_id in ordered_ids
            ),
        },
        "conservative_consensus": {
            "rule": "minimum relevance label from the two reviewers; not a substitute for adjudication",
            "metrics": _single_system_metrics(conservative, manifest_rows),
            "system_metrics": _retrieval_metrics_by_system(conservative, manifest_rows),
            "system_top1_comparison": _compare_system_top1(conservative, manifest_rows),
        },
        "adjudication": {
            "required": bool(disagreements),
            "row_count": len(disagreements),
            "note": "Do not average ordinal labels. Resolve each disagreement against the rubric.",
        },
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
    }
    return report, disagreements


def write_analysis(
    report: dict[str, Any],
    disagreements: list[dict[str, Any]],
    *,
    report_path: Path,
    adjudication_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    adjudication_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(disagreements[0]) if disagreements else ["row_id", "adjudicated_relevance", "adjudication_notes"]
    with adjudication_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(disagreements)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reviewer-a-name", default="reviewer_1")
    parser.add_argument("--reviewer-b-name", default="reviewer_2")
    parser.add_argument("--report", required=True)
    parser.add_argument("--adjudication", required=True)
    args = parser.parse_args()
    report, disagreements = analyze_reviews(
        reviewer_a_path=Path(args.reviewer_a),
        reviewer_b_path=Path(args.reviewer_b),
        manifest_path=Path(args.manifest),
        reviewer_a_name=args.reviewer_a_name,
        reviewer_b_name=args.reviewer_b_name,
    )
    write_analysis(
        report,
        disagreements,
        report_path=Path(args.report),
        adjudication_path=Path(args.adjudication),
    )
    print(json.dumps({"ok": True, "agreement": report["agreement"], "adjudication_rows": len(disagreements)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
