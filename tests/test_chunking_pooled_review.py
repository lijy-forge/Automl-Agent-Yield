from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_yieldmind_retrieval_reviews import analyze_reviews
from scripts.prepare_yieldmind_chunking_review import prepare_pooled_review


def _report(path: Path, candidates: list[dict[str, object]]) -> None:
    payload = {
        "baselines": {
            "hybrid": {
                "cases": [
                    {
                        "id": "case_1",
                        "query": "Which candidate directly answers the question?",
                        "candidates": candidates,
                    }
                ]
            }
        },
        "ingestion": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(rank: int, source_path: str, chunk_id: str) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "source_path": source_path,
        "chunk_index": 0,
        "retrieval_channels": ["vector", "bm25"],
        "retrieval_channel_ranks": {"vector": rank, "bm25": rank},
    }


def _label_review(path: Path, labels_by_text: dict[str, str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        row["relevance"] = labels_by_text[row["candidate_text"]]
        row["confidence"] = "high"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_pooled_review_hides_systems_and_scores_shared_gold(tmp_path: Path) -> None:
    sources = tmp_path / "knowledge_sources"
    sources.mkdir()
    (sources / "shared.md").write_text("Direct answer.", encoding="utf-8")
    (sources / "only_a.md").write_text("Background A.", encoding="utf-8")
    (sources / "only_b.md").write_text("Background B.", encoding="utf-8")
    report_a = tmp_path / "a.json"
    report_b = tmp_path / "b.json"
    _report(
        report_a,
        [
            _candidate(1, "knowledge_sources/shared.md", "a-shared"),
            _candidate(2, "knowledge_sources/only_a.md", "a-only"),
        ],
    )
    _report(
        report_b,
        [
            _candidate(1, "knowledge_sources/only_b.md", "b-only"),
            _candidate(2, "knowledge_sources/shared.md", "b-shared"),
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "system_id": "v3",
                        "report_path": str(report_a),
                        "retrieval_mode": "hybrid",
                        "split_strategy": "recursive_chars_v3",
                        "chunk_size": 100,
                        "chunk_overlap": 10,
                    },
                    {
                        "system_id": "v4",
                        "report_path": str(report_b),
                        "retrieval_mode": "hybrid",
                        "split_strategy": "section_aware_v4",
                        "chunk_size": 100,
                        "chunk_overlap": 10,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "review"
    manifest_path = tmp_path / "manifest.json"
    result = prepare_pooled_review(
        config_path=config,
        sources_root=sources,
        output_dir=output_dir,
        manifest_path=manifest_path,
        instructions_path=output_dir / "README.md",
        reviewer_ids=["reviewer_1", "reviewer_2"],
        seed=7,
    )

    assert result["row_count"] == 3
    assert result["candidate_membership_count"] == 4
    assert result["shared_candidate_count"] == 1
    first = output_dir / "yieldmind_chunking_review_reviewer_1.csv"
    second = output_dir / "yieldmind_chunking_review_reviewer_2.csv"
    public_text = first.read_text(encoding="utf-8-sig")
    assert "v3" not in public_text
    assert "v4" not in public_text
    assert "retrieval_rank" not in public_text

    labels = {"Direct answer.": "2", "Background A.": "0", "Background B.": "0"}
    _label_review(first, labels)
    _label_review(second, labels)
    report, disagreements = analyze_reviews(
        reviewer_a_path=first,
        reviewer_b_path=second,
        manifest_path=manifest_path,
        reviewer_a_name="reviewer_1",
        reviewer_b_name="reviewer_2",
    )

    assert disagreements == []
    assert report["conservative_consensus"]["metrics"] == {}
    system_metrics = report["conservative_consensus"]["system_metrics"]
    assert system_metrics["v3"]["mrr_direct"] == 1.0
    assert system_metrics["v4"]["mrr_direct"] == 0.5
    comparison = report["conservative_consensus"]["system_top1_comparison"]
    assert comparison["wins"] == {"v3": 1, "v4": 0}
    assert comparison["ties"] == 0
    assert comparison["top1_direct_delta_second_minus_first"] == -1.0

    top1_result = prepare_pooled_review(
        config_path=config,
        sources_root=sources,
        output_dir=tmp_path / "top1_review",
        manifest_path=tmp_path / "top1_manifest.json",
        instructions_path=tmp_path / "top1_review" / "README.md",
        reviewer_ids=["reviewer_1", "reviewer_2"],
        seed=7,
        candidate_scope="top1_union",
    )
    assert top1_result["candidate_scope"] == "top1_union"
    assert top1_result["row_count"] == 2
    top1_manifest = json.loads((tmp_path / "top1_manifest.json").read_text(encoding="utf-8"))
    assert top1_manifest["candidate_scope"] == "top1_union"
    assert all(
        any(int(membership["retrieval_rank"]) == 1 for membership in row["systems"].values())
        for row in top1_manifest["rows"]
    )


def test_pooled_review_rejects_report_config_split_mismatch(tmp_path: Path) -> None:
    sources = tmp_path / "knowledge_sources"
    sources.mkdir()
    (sources / "shared.md").write_text("Direct answer.", encoding="utf-8")
    report_a = tmp_path / "a.json"
    report_b = tmp_path / "b.json"
    candidates = [_candidate(1, "knowledge_sources/shared.md", "shared")]
    _report(report_a, candidates)
    _report(report_b, candidates)
    payload = json.loads(report_a.read_text(encoding="utf-8"))
    payload["ingestion"]["split_version"] = "recursive_chars_1200_180_v1"
    report_a.write_text(json.dumps(payload), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "system_id": system_id,
                        "report_path": str(report),
                        "retrieval_mode": "hybrid",
                        "split_strategy": "recursive_chars_v3",
                        "chunk_size": 700,
                        "chunk_overlap": 80,
                    }
                    for system_id, report in (("bad", report_a), ("good", report_b))
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split mismatch"):
        prepare_pooled_review(
            config_path=config,
            sources_root=sources,
            output_dir=tmp_path / "review",
            manifest_path=tmp_path / "manifest.json",
            instructions_path=tmp_path / "README.md",
            reviewer_ids=["reviewer_1", "reviewer_2"],
            seed=7,
        )
