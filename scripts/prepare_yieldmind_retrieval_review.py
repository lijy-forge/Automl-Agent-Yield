#!/usr/bin/env python3
"""Create a blind CSV review pack from retrieval candidate diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import json_dumps
from yieldmind.knowledge_base import split_knowledge_text, split_version_for


class RetrievalReviewRow(BaseModel):
    reviewer_id: str = Field(min_length=1)
    row_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    query_zh: str = ""
    candidate_code: str = Field(pattern=r"^[A-Z]$")
    candidate_text: str = Field(min_length=1)
    candidate_text_zh: str = ""
    relevance: Literal["", "0", "1", "2"] = ""
    confidence: Literal["", "low", "medium", "high"] = ""
    notes: str = ""


CSV_FIELDS = list(RetrievalReviewRow.model_fields)
REVIEWER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _source_file(source_path: str, sources_root: Path) -> Path:
    relative = Path(source_path)
    if relative.is_absolute():
        raise ValueError("Review source paths must be repository-relative.")
    if relative.parts and relative.parts[0] == sources_root.name:
        relative = Path(*relative.parts[1:])
    resolved = (sources_root / relative).resolve()
    try:
        resolved.relative_to(sources_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Review source escapes the sources directory: {source_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Review source does not exist: {resolved}")
    return resolved


def prepare_review_pack(
    *,
    retrieval_report: Path,
    sources_root: Path,
    output_csv: Path,
    manifest_path: Path,
    instructions_path: Path,
    reviewer_id: str,
    retrieval_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    seed: int,
    translations: dict[str, str] | None = None,
    query_translations: dict[str, str] | None = None,
    overwrite_unlabeled: bool = False,
) -> dict[str, Any]:
    if not REVIEWER_ID_RE.fullmatch(reviewer_id):
        raise ValueError("reviewer_id must contain only letters, numbers, underscores, or hyphens.")
    if output_csv.exists():
        if not overwrite_unlabeled:
            raise FileExistsError(f"Review CSV already exists and will not be overwritten: {output_csv}")
        with output_csv.open(encoding="utf-8-sig", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        editable_fields = ("relevance", "confidence", "notes")
        if any(any(str(row.get(field, "")).strip() for field in editable_fields) for row in existing_rows):
            raise ValueError("Existing review CSV contains labels or notes and cannot be overwritten.")
    report_bytes = retrieval_report.read_bytes()
    report = json.loads(report_bytes)
    baseline = report.get("baselines", {}).get(retrieval_mode)
    if not isinstance(baseline, dict) or not isinstance(baseline.get("cases"), list):
        raise ValueError(f"Retrieval report has no candidate diagnostics for mode {retrieval_mode!r}.")

    cases = list(baseline["cases"])
    random.Random(seed).shuffle(cases)
    chunks_by_source: dict[str, list[str]] = {}
    review_rows: list[RetrievalReviewRow] = []
    manifest_rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        query_zh = ""
        if query_translations is not None:
            query_zh = str(query_translations.get(case_id) or "")
            if not query_zh:
                raise ValueError(f"Missing Chinese translation for query {case_id!r}.")
        candidates = list(case.get("candidates") or [])
        if not candidates:
            raise ValueError(f"Case {case.get('id')!r} has no candidate diagnostics.")
        random.Random(f"{seed}:{case['id']}").shuffle(candidates)
        for offset, candidate in enumerate(candidates):
            candidate_code = chr(ord("A") + offset)
            source_path = str(candidate.get("source_path") or "")
            source_file = _source_file(source_path, sources_root)
            if source_path not in chunks_by_source:
                text = source_file.read_text(encoding="utf-8")
                chunks_by_source[source_path] = split_knowledge_text(
                    text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            chunk_index = int(candidate["chunk_index"])
            chunks = chunks_by_source[source_path]
            if chunk_index < 0 or chunk_index >= len(chunks):
                raise ValueError(
                    f"Candidate chunk index {chunk_index} is invalid for {source_path}; found {len(chunks)} chunks."
                )
            candidate_text = chunks[chunk_index]
            text_sha256 = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
            candidate_text_zh = ""
            if translations is not None:
                candidate_text_zh = str(translations.get(text_sha256) or translations.get(text_sha256[:12]) or "")
                if not candidate_text_zh:
                    raise ValueError(f"Missing Chinese translation for candidate text {text_sha256[:12]}.")
            row_id = f"{case_id}:{candidate_code}"
            row = RetrievalReviewRow(
                reviewer_id=reviewer_id,
                row_id=row_id,
                case_id=case_id,
                query=str(case["query"]),
                query_zh=query_zh,
                candidate_code=candidate_code,
                candidate_text=candidate_text,
                candidate_text_zh=candidate_text_zh,
            )
            review_rows.append(row)
            manifest_rows.append(
                {
                    "row_id": row_id,
                    "case_id": case["id"],
                    "candidate_code": candidate_code,
                    "chunk_id": candidate.get("chunk_id"),
                    "source_path": source_path,
                    "chunk_index": chunk_index,
                    "text_sha256": text_sha256,
                    "retrieval_rank": candidate.get("rank"),
                    "retrieval_channels": candidate.get("retrieval_channels", []),
                    "retrieval_channel_ranks": candidate.get("retrieval_channel_ranks", {}),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.model_dump() for row in review_rows)

    manifest = {
        "version": "yieldmind-retrieval-blind-review-v1",
        "reviewer_id": reviewer_id,
        "retrieval_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "retrieval_mode": retrieval_mode,
        "split_version": split_version_for(chunk_size, chunk_overlap),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "seed": seed,
        "case_count": len(cases),
        "row_count": len(review_rows),
        "translation_language": "zh-CN" if translations is not None or query_translations is not None else "",
        "machine_translation_assistance": translations is not None or query_translations is not None,
        "rows": manifest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json_dumps(manifest) + "\n", encoding="utf-8")

    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    instructions_path.write_text(
        "# YieldMind Retrieval Blind Review\n\n"
        "只编辑 CSV 中的 `relevance`、`confidence` 和 `notes` 三列。不要查看本地 manifest。\n"
        "`query` / `candidate_text` 是英文原文，`query_zh` / `candidate_text_zh` 是机器生成的中文辅助翻译；"
        "有冲突时以英文原文为准。\n\n"
        "- `2`：候选文本直接回答问题，可单独作为该问题的主要引用。\n"
        "- `1`：候选文本提供相关背景或部分答案，但不足以单独支撑完整结论。\n"
        "- `0`：候选文本无关、答非所问，或可能误导回答。\n"
        "- `confidence`：填写 `high`、`medium` 或 `low`。\n"
        "- `notes`：可留空；边界模糊、问题本身有歧义或需要多个候选组合时请说明。\n\n"
        "逐行根据 `query`、英文原文和中文辅助翻译判断。不要根据候选编号猜测检索排名，"
        "不要修改 `reviewer_id`、`row_id`、`case_id`、`query`、`query_zh`、`candidate_code`、英文或中文文本。\n"
        "完成前确认每一行的 `relevance` 和 `confidence` 都已填写。CSV 使用 UTF-8 BOM，可直接用 Excel 打开。\n",
        encoding="utf-8",
    )
    return {
        "status": "prepared",
        "reviewer_id": reviewer_id,
        "case_count": len(cases),
        "row_count": len(review_rows),
        "output_csv": str(output_csv),
        "instructions": str(instructions_path),
        "manifest": str(manifest_path),
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--reviewer-id", default="reviewer_1")
    parser.add_argument("--retrieval-mode", choices=("vector", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--translations", default="")
    parser.add_argument("--overwrite-unlabeled", action="store_true")
    parser.add_argument("--sources-root", default=str(PROJECT_ROOT / "knowledge_sources"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "evals" / "review"))
    parser.add_argument(
        "--manifest-dir",
        default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "retrieval_reviews"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    translations = None
    query_translations = None
    if args.translations:
        translation_payload = json.loads(Path(args.translations).read_text(encoding="utf-8"))
        translations = translation_payload.get("translations")
        query_translations = translation_payload.get("query_translations")
        if not isinstance(translations, dict):
            raise ValueError("Translation file must contain a translations object.")
        if not isinstance(query_translations, dict):
            raise ValueError("Translation file must contain a query_translations object.")
    result = prepare_review_pack(
        retrieval_report=Path(args.retrieval_report),
        sources_root=Path(args.sources_root),
        output_csv=output_dir / f"yieldmind_retrieval_review_{args.reviewer_id}.csv",
        manifest_path=Path(args.manifest_dir) / f"yieldmind_retrieval_review_{args.reviewer_id}_manifest.json",
        instructions_path=output_dir / "README.md",
        reviewer_id=args.reviewer_id,
        retrieval_mode=args.retrieval_mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        seed=args.seed,
        translations=translations,
        query_translations=query_translations,
        overwrite_unlabeled=args.overwrite_unlabeled,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
