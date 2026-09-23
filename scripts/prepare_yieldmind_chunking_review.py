#!/usr/bin/env python3
"""Pool multiple retrieval systems into one system-blind review pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_yieldmind_retrieval_review import CSV_FIELDS, RetrievalReviewRow, _source_file
from yieldmind.database import json_dumps
from yieldmind.document_loader import load_document
from yieldmind.knowledge_base import SplitStrategy, split_knowledge_section, split_version_for


SYSTEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,60}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _system_chunks(
    *,
    source_path: str,
    sources_root: Path,
    split_strategy: SplitStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    loaded = load_document(_source_file(source_path, sources_root))
    return [
        draft.text
        for section in loaded.sections
        for draft in split_knowledge_section(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_strategy=split_strategy,
        )
    ]


def _validate_system(raw: dict[str, Any]) -> dict[str, Any]:
    system_id = str(raw.get("system_id") or "")
    if not SYSTEM_ID_RE.fullmatch(system_id):
        raise ValueError(f"Invalid system_id {system_id!r}.")
    retrieval_mode = str(raw.get("retrieval_mode") or "hybrid")
    if retrieval_mode not in {"vector", "bm25", "hybrid"}:
        raise ValueError(f"Invalid retrieval_mode for {system_id!r}: {retrieval_mode!r}.")
    split_strategy = str(raw.get("split_strategy") or "")
    if split_strategy not in {"recursive_chars_v3", "section_aware_v4"}:
        raise ValueError(f"Invalid split_strategy for {system_id!r}: {split_strategy!r}.")
    chunk_size = int(raw.get("chunk_size") or 0)
    chunk_overlap = int(raw.get("chunk_overlap") or 0)
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(f"Invalid chunk configuration for {system_id!r}.")
    report_path = Path(str(raw.get("report_path") or "")).expanduser()
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Retrieval report does not exist for {system_id!r}: {report_path}")
    return {
        "system_id": system_id,
        "retrieval_mode": retrieval_mode,
        "split_strategy": split_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "report_path": report_path,
    }


def _write_review_csv(path: Path, reviewer_id: str, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if any(
            str(row.get(field, "")).strip()
            for row in existing
            for field in ("relevance", "confidence", "notes")
        ):
            raise ValueError(f"Existing review CSV contains review work and cannot be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for raw in rows:
            writer.writerow(RetrievalReviewRow(reviewer_id=reviewer_id, **raw).model_dump())


def prepare_pooled_review(
    *,
    config_path: Path,
    sources_root: Path,
    output_dir: Path,
    manifest_path: Path,
    instructions_path: Path,
    reviewer_ids: list[str],
    seed: int,
    candidate_scope: str = "full_pool",
) -> dict[str, Any]:
    if candidate_scope not in {"full_pool", "top1_union"}:
        raise ValueError(f"Unsupported candidate_scope: {candidate_scope!r}.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_systems = config.get("systems")
    if not isinstance(raw_systems, list) or len(raw_systems) < 2:
        raise ValueError("Pooling config must contain at least two systems.")
    systems = [_validate_system(item) for item in raw_systems]
    system_ids = [item["system_id"] for item in systems]
    if len(system_ids) != len(set(system_ids)):
        raise ValueError("Pooling config has duplicate system_id values.")

    reports: dict[str, dict[str, Any]] = {}
    case_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for system in systems:
        report_bytes = system["report_path"].read_bytes()
        report = json.loads(report_bytes)
        expected_split_version = split_version_for(
            system["chunk_size"], system["chunk_overlap"], system["split_strategy"]
        )
        reported_split_version = str(report.get("ingestion", {}).get("split_version") or "")
        if reported_split_version and reported_split_version != expected_split_version:
            raise ValueError(
                f"Report/config split mismatch for {system['system_id']!r}: "
                f"reported {reported_split_version!r}, expected {expected_split_version!r}."
            )
        baseline = report.get("baselines", {}).get(system["retrieval_mode"])
        if not isinstance(baseline, dict) or not isinstance(baseline.get("cases"), list):
            raise ValueError(
                f"Report for {system['system_id']!r} has no {system['retrieval_mode']!r} cases."
            )
        reports[system["system_id"]] = {
            "bytes": report_bytes,
            "report": report,
            "baseline": baseline,
        }
        cases = {str(case["id"]): case for case in baseline["cases"]}
        if len(cases) != len(baseline["cases"]):
            raise ValueError(f"Report for {system['system_id']!r} has duplicate case IDs.")
        case_sets[system["system_id"]] = cases

    reference_id = system_ids[0]
    reference_cases = case_sets[reference_id]
    for system_id in system_ids[1:]:
        if set(case_sets[system_id]) != set(reference_cases):
            raise ValueError(f"Case IDs differ between {reference_id!r} and {system_id!r}.")
        for case_id, reference in reference_cases.items():
            if str(case_sets[system_id][case_id].get("query")) != str(reference.get("query")):
                raise ValueError(f"Query differs for case {case_id!r} between pooled systems.")

    chunk_cache: dict[tuple[str, str, int, int], list[str]] = {}
    pooled_cases: list[dict[str, Any]] = []
    membership_count = 0
    for case_id, reference_case in reference_cases.items():
        candidates_by_hash: dict[str, dict[str, Any]] = {}
        for system in systems:
            system_id = system["system_id"]
            candidates = list(case_sets[system_id][case_id].get("candidates") or [])
            if not candidates:
                raise ValueError(f"Case {case_id!r} has no candidates for system {system_id!r}.")
            for candidate in candidates:
                source_path = str(candidate.get("source_path") or "")
                cache_key = (
                    source_path,
                    system["split_strategy"],
                    system["chunk_size"],
                    system["chunk_overlap"],
                )
                if cache_key not in chunk_cache:
                    chunk_cache[cache_key] = _system_chunks(
                        source_path=source_path,
                        sources_root=sources_root,
                        split_strategy=system["split_strategy"],
                        chunk_size=system["chunk_size"],
                        chunk_overlap=system["chunk_overlap"],
                    )
                chunk_index = int(candidate["chunk_index"])
                chunks = chunk_cache[cache_key]
                if chunk_index < 0 or chunk_index >= len(chunks):
                    raise ValueError(
                        f"Invalid chunk_index {chunk_index} for {source_path!r} in {system_id!r}; "
                        f"reconstructed {len(chunks)} chunks."
                    )
                candidate_text = chunks[chunk_index]
                text_sha256 = _sha256(candidate_text.encode("utf-8"))
                pooled = candidates_by_hash.setdefault(
                    text_sha256,
                    {
                        "candidate_text": candidate_text,
                        "text_sha256": text_sha256,
                        "systems": {},
                    },
                )
                rank = int(candidate["rank"])
                previous = pooled["systems"].get(system_id)
                if previous is None or rank < int(previous["retrieval_rank"]):
                    pooled["systems"][system_id] = {
                        "retrieval_rank": rank,
                        "chunk_id": candidate.get("chunk_id"),
                        "source_path": source_path,
                        "chunk_index": chunk_index,
                        "retrieval_channels": candidate.get("retrieval_channels", []),
                        "retrieval_channel_ranks": candidate.get("retrieval_channel_ranks", {}),
                    }
        pooled_candidates = list(candidates_by_hash.values())
        if candidate_scope == "top1_union":
            pooled_candidates = [
                {
                    **candidate,
                    "systems": {
                        system_id: membership
                        for system_id, membership in candidate["systems"].items()
                        if int(membership["retrieval_rank"]) == 1
                    },
                }
                for candidate in pooled_candidates
                if any(
                    int(membership["retrieval_rank"]) == 1
                    for membership in candidate["systems"].values()
                )
            ]
        if len(pooled_candidates) > 26:
            raise ValueError(f"Case {case_id!r} has {len(pooled_candidates)} pooled candidates; maximum is 26.")
        membership_count += sum(len(item["systems"]) for item in pooled_candidates)
        random.Random(f"{seed}:{case_id}").shuffle(pooled_candidates)
        pooled_cases.append(
            {
                "case_id": case_id,
                "query": str(reference_case["query"]),
                "candidates": pooled_candidates,
            }
        )
    random.Random(seed).shuffle(pooled_cases)

    public_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for case in pooled_cases:
        for offset, candidate in enumerate(case["candidates"]):
            code = chr(ord("A") + offset)
            row_id = f"{case['case_id']}:{code}"
            public_rows.append(
                {
                    "row_id": row_id,
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "query_zh": "",
                    "candidate_code": code,
                    "candidate_text": candidate["candidate_text"],
                    "candidate_text_zh": "",
                }
            )
            representative = next(iter(candidate["systems"].values()))
            manifest_rows.append(
                {
                    "row_id": row_id,
                    "case_id": case["case_id"],
                    "candidate_code": code,
                    "text_sha256": candidate["text_sha256"],
                    "pool_display_rank": offset + 1,
                    "retrieval_rank": offset + 1,
                    "chunk_id": representative.get("chunk_id"),
                    "source_path": representative.get("source_path"),
                    "chunk_index": representative.get("chunk_index"),
                    "systems": candidate["systems"],
                }
            )

    output_files: dict[str, str] = {}
    for reviewer_id in reviewer_ids:
        if not SYSTEM_ID_RE.fullmatch(reviewer_id):
            raise ValueError(f"Invalid reviewer_id {reviewer_id!r}.")
        pack_name = "yieldmind_chunking_top1_review" if candidate_scope == "top1_union" else "yieldmind_chunking_review"
        output_path = output_dir / f"{pack_name}_{reviewer_id}.csv"
        _write_review_csv(output_path, reviewer_id, public_rows)
        output_files[reviewer_id] = str(output_path)

    manifest_systems = {}
    for system in systems:
        system_id = system["system_id"]
        report = reports[system_id]["report"]
        manifest_systems[system_id] = {
            "report_file": system["report_path"].name,
            "report_sha256": _sha256(reports[system_id]["bytes"]),
            "retrieval_mode": system["retrieval_mode"],
            "split_strategy": system["split_strategy"],
            "chunk_size": system["chunk_size"],
            "chunk_overlap": system["chunk_overlap"],
            "split_version": split_version_for(
                system["chunk_size"], system["chunk_overlap"], system["split_strategy"]
            ),
            "reported_split_version": report.get("ingestion", {}).get("split_version"),
        }
    manifest = {
        "version": "yieldmind-chunking-pooled-blind-review-v1",
        "config_file": config_path.name,
        "config_sha256": _sha256(config_path.read_bytes()),
        "seed": seed,
        "candidate_scope": candidate_scope,
        "case_count": len(pooled_cases),
        "row_count": len(manifest_rows),
        "candidate_membership_count": membership_count,
        "shared_candidate_count": sum(len(row["systems"]) > 1 for row in manifest_rows),
        "systems": manifest_systems,
        "rows": manifest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")

    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    instructions_path.write_text(
        "# YieldMind 分块 A/B 池化盲审\n\n"
        + (
            "本包是 Top-1 快速筛选：只评审两套系统首位候选的并集，只能比较 Top-1，"
            "不能用来声称完整 Recall@5、MRR 或 nDCG。\n\n"
            if candidate_scope == "top1_union"
            else ""
        )
        +
        "两套分块的 Top-K 候选已按文本合并去重，系统身份和原始排名已隐藏。"
        "两位评审独立完成各自 CSV，不要互相参考。\n\n"
        "只编辑 `relevance`、`confidence` 和 `notes`：\n\n"
        "- `2`：候选文本直接回答问题，可单独作为主要引用。\n"
        "- `1`：提供相关背景或部分答案，但不足以单独支撑完整结论。\n"
        "- `0`：无关、答非所问，或可能误导回答。\n"
        "- `confidence`：`high`、`medium` 或 `low`。\n"
        "- `notes`：可留空；边界模糊、问题有歧义或需多片段组合时请说明。\n\n"
        "不要修改其他列，不要查看私有 manifest。CSV 使用 UTF-8 BOM。\n",
        encoding="utf-8",
    )
    return {
        "status": "prepared",
        "candidate_scope": candidate_scope,
        "case_count": len(pooled_cases),
        "row_count": len(manifest_rows),
        "candidate_membership_count": membership_count,
        "shared_candidate_count": manifest["shared_candidate_count"],
        "review_files": output_files,
        "manifest": str(manifest_path),
        "instructions": str(instructions_path),
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reviewer-id", action="append", dest="reviewer_ids")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--candidate-scope",
        choices=("full_pool", "top1_union"),
        default="full_pool",
    )
    parser.add_argument("--sources-root", default=str(PROJECT_ROOT / "knowledge_sources"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "evals" / "review"))
    parser.add_argument(
        "--manifest",
        default=str(
            PROJECT_ROOT
            / "agent_workspace"
            / "yieldmind"
            / "retrieval_reviews"
            / "yieldmind_chunking_pooled_manifest.json"
        ),
    )
    parser.add_argument(
        "--instructions",
        default=str(PROJECT_ROOT / "evals" / "review" / "README_chunking_pool.md"),
    )
    args = parser.parse_args()
    result = prepare_pooled_review(
        config_path=Path(args.config),
        sources_root=Path(args.sources_root),
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
        instructions_path=Path(args.instructions),
        reviewer_ids=args.reviewer_ids or ["reviewer_1", "reviewer_2"],
        seed=args.seed,
        candidate_scope=args.candidate_scope,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
