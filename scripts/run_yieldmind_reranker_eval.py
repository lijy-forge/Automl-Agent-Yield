#!/usr/bin/env python3
"""Replay the adjudicated Top-5 retrieval set through a pinned Qwen3 reranker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GOLD = PROJECT_ROOT / "evals" / "review" / "yieldmind_retrieval_gold.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "reranker_evals"
DEFAULT_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_MODEL_REVISION = "204631fece3ea330da97bf0175d0a907d910df20"
EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_QUERY_INSTRUCTION = (
    "Instruct: Given a query about yield-stress modeling and the YieldMind system, "
    "retrieve relevant passages that answer the query\nQuery:"
)
DEFAULT_INSTRUCTION = (
    "Given a query about yield-stress modeling and the YieldMind system, "
    "retrieve passages that directly answer the query."
)
REQUIRED_FIELDS = {
    "row_id",
    "case_id",
    "query",
    "candidate_text",
    "chunk_id",
    "source_path",
    "retrieval_rank",
    "final_relevance",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Gold file has no header: {path}")
        missing = REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Gold file is missing columns: {sorted(missing)}")
        raw_rows = [dict(row) for row in reader]
    if not raw_rows:
        raise ValueError(f"Gold file is empty: {path}")

    row_ids = [row["row_id"].strip() for row in raw_rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Gold file contains duplicate row IDs.")

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        try:
            rank = int(raw["retrieval_rank"])
            relevance = int(raw["final_relevance"])
        except ValueError as exc:
            raise ValueError(f"Gold row has a non-integer rank or relevance: {raw['row_id']}") from exc
        if relevance not in {0, 1, 2}:
            raise ValueError(f"Gold row has invalid relevance: {raw['row_id']}")
        if rank <= 0 or not raw["candidate_text"].strip() or not raw["query"].strip():
            raise ValueError(f"Gold row has invalid rank or blank text: {raw['row_id']}")
        row: dict[str, Any] = {**raw, "retrieval_rank": rank, "final_relevance": relevance}
        rows.append(row)
        grouped[row["case_id"]].append(row)

    for case_id, candidates in grouped.items():
        queries = {row["query"] for row in candidates}
        ranks = sorted(int(row["retrieval_rank"]) for row in candidates)
        if len(queries) != 1:
            raise ValueError(f"Gold case has inconsistent queries: {case_id}")
        if ranks != list(range(1, len(candidates) + 1)):
            raise ValueError(f"Gold case ranks must be contiguous and unique: {case_id}")
    return rows


def _ranking_metrics(grouped_ranked_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    reciprocal_direct: list[float] = []
    reciprocal_partial: list[float] = []
    ndcg: list[float] = []
    cases_without_direct: list[str] = []
    top_labels: list[int] = []
    for case_id in sorted(grouped_ranked_rows):
        rows = grouped_ranked_rows[case_id]
        labels = [int(row["final_relevance"]) for row in rows]
        top_labels.append(labels[0])
        direct_ranks = [rank for rank, label in enumerate(labels, start=1) if label >= 2]
        partial_ranks = [rank for rank, label in enumerate(labels, start=1) if label >= 1]
        reciprocal_direct.append(1.0 / direct_ranks[0] if direct_ranks else 0.0)
        reciprocal_partial.append(1.0 / partial_ranks[0] if partial_ranks else 0.0)
        if not direct_ranks:
            cases_without_direct.append(case_id)
        dcg = sum((2**label - 1) / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))
        ideal = sorted(labels, reverse=True)
        ideal_dcg = sum((2**label - 1) / math.log2(rank + 1) for rank, label in enumerate(ideal, start=1))
        ndcg.append(dcg / ideal_dcg if ideal_dcg else 0.0)

    case_count = len(grouped_ranked_rows)
    return {
        "case_count": case_count,
        "recall_at_5_direct": (case_count - len(cases_without_direct)) / case_count,
        "mrr_direct": sum(reciprocal_direct) / case_count,
        "mrr_partial": sum(reciprocal_partial) / case_count,
        "top1_direct": sum(label >= 2 for label in top_labels) / case_count,
        "top1_partial": sum(label >= 1 for label in top_labels) / case_count,
        "ndcg_at_5": sum(ndcg) / case_count,
        "cases_without_direct": cases_without_direct,
    }


def evaluate_reranker_scores(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != len(scores):
        raise ValueError(f"Expected {len(rows)} reranker scores; received {len(scores)}.")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("Reranker scores must all be finite.")

    original: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reranked: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        scored = {**row, "reranker_score": float(score)}
        original[str(row["case_id"])].append(scored)
        reranked[str(row["case_id"])].append(scored)
    for case_id in original:
        original[case_id].sort(key=lambda row: int(row["retrieval_rank"]))
        reranked[case_id].sort(key=lambda row: (-float(row["reranker_score"]), int(row["retrieval_rank"])))

    original_metrics = _ranking_metrics(original)
    reranked_metrics = _ranking_metrics(reranked)
    metric_deltas = {
        key: reranked_metrics[key] - original_metrics[key]
        for key in ("recall_at_5_direct", "mrr_direct", "mrr_partial", "top1_direct", "top1_partial", "ndcg_at_5")
    }
    case_results: list[dict[str, Any]] = []
    direct_promoted = 0
    direct_demoted = 0
    top1_changed = 0
    for case_id in sorted(original):
        before = original[case_id]
        after = reranked[case_id]
        before_direct = int(before[0]["final_relevance"]) >= 2
        after_direct = int(after[0]["final_relevance"]) >= 2
        top1_changed += before[0]["row_id"] != after[0]["row_id"]
        direct_promoted += not before_direct and after_direct
        direct_demoted += before_direct and not after_direct
        case_results.append(
            {
                "case_id": case_id,
                "query": before[0]["query"],
                "original_top1_row_id": before[0]["row_id"],
                "original_top1_relevance": before[0]["final_relevance"],
                "reranked_top1_row_id": after[0]["row_id"],
                "reranked_top1_relevance": after[0]["final_relevance"],
                "top1_changed": before[0]["row_id"] != after[0]["row_id"],
                "ranking": [
                    {
                        "reranked_rank": rank,
                        "original_rank": row["retrieval_rank"],
                        "row_id": row["row_id"],
                        "chunk_id": row["chunk_id"],
                        "source_path": row["source_path"],
                        "final_relevance": row["final_relevance"],
                        "reranker_score": row["reranker_score"],
                    }
                    for rank, row in enumerate(after, start=1)
                ],
            }
        )
    comparison = {
        "original_metrics": original_metrics,
        "reranked_metrics": reranked_metrics,
        "metric_deltas": metric_deltas,
        "top1_changed_count": top1_changed,
        "direct_relevance_promoted_to_top1": direct_promoted,
        "direct_relevance_demoted_from_top1": direct_demoted,
    }
    return comparison, case_results


class Qwen3Reranker:
    """Minimal implementation of the official yes/no causal-LM scoring protocol."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        instruction: str,
        max_length: int,
        batch_size: int,
        device: str,
        allow_model_download: bool,
    ) -> None:
        if not revision or revision.lower() == "main":
            raise ValueError("Reranker revision must be an immutable commit, not 'main'.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        load_started = time.perf_counter()
        load_kwargs = {
            "revision": revision,
            "local_files_only": not allow_model_download,
            "trust_remote_code": True,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", **load_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.load_duration_seconds = time.perf_counter() - load_started
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.false_token_id == self.tokenizer.unk_token_id or self.true_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Reranker tokenizer does not expose the required yes/no tokens.")
        prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
            'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def score(self, pairs: Sequence[tuple[str, str]]) -> tuple[list[float], dict[str, Any]]:
        formatted = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}" for query, document in pairs]
        scores: list[float] = []
        batch_latencies_ms: list[float] = []
        token_count = 0
        inference_started = time.perf_counter()
        with self.torch.no_grad():
            for offset in range(0, len(formatted), self.batch_size):
                batch = formatted[offset : offset + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
                    return_attention_mask=False,
                )
                inputs["input_ids"] = [
                    self.prefix_tokens + token_ids + self.suffix_tokens for token_ids in inputs["input_ids"]
                ]
                padded = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
                token_count += int(padded["attention_mask"].sum().item())
                padded = {key: value.to(self.device) for key, value in padded.items()}
                batch_started = time.perf_counter()
                logits = self.model(**padded).logits[:, -1, :]
                binary = self.torch.stack(
                    [logits[:, self.false_token_id], logits[:, self.true_token_id]], dim=1
                )
                batch_scores = self.torch.nn.functional.log_softmax(binary, dim=1)[:, 1].exp()
                if self.device == "mps":
                    self.torch.mps.synchronize()
                elif self.device == "cuda":
                    self.torch.cuda.synchronize()
                batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)
                scores.extend(float(value) for value in batch_scores.detach().float().cpu().tolist())
        duration = time.perf_counter() - inference_started
        telemetry = {
            "pair_count": len(pairs),
            "batch_count": len(batch_latencies_ms),
            "batch_size": self.batch_size,
            "input_token_count": token_count,
            "model_load_duration_seconds": round(self.load_duration_seconds, 4),
            "inference_duration_seconds": round(duration, 4),
            "mean_pair_latency_ms": round(duration * 1000.0 / max(1, len(pairs)), 4),
            "mean_batch_latency_ms": round(sum(batch_latencies_ms) / max(1, len(batch_latencies_ms)), 4),
            "max_batch_latency_ms": round(max(batch_latencies_ms, default=0.0), 4),
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
        }
        return scores, telemetry


class Qwen3EmbeddingCosineScorer:
    """Second-stage dense-score ablation using the already validated embedding model."""

    def __init__(
        self,
        *,
        max_length: int,
        batch_size: int,
        device: str,
        allow_model_download: bool,
    ) -> None:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.batch_size = batch_size
        if device == "auto":
            device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        load_started = time.perf_counter()
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_ID,
            revision=EMBEDDING_MODEL_REVISION,
            local_files_only=not allow_model_download,
            tokenizer_kwargs={"padding_side": "left"},
            device=device,
        )
        self.model.max_seq_length = max_length
        self.load_duration_seconds = time.perf_counter() - load_started
        self.device = str(self.model.device)

    def score(self, pairs: Sequence[tuple[str, str]]) -> tuple[list[float], dict[str, Any]]:
        unique_queries = list(dict.fromkeys(query for query, _ in pairs))
        unique_documents = list(dict.fromkeys(document for _, document in pairs))
        query_inputs = [f"{EMBEDDING_QUERY_INSTRUCTION}{query}" for query in unique_queries]
        started = time.perf_counter()
        query_vectors = self.model.encode(
            query_inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        document_vectors = self.model.encode(
            unique_documents,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        duration = time.perf_counter() - started
        query_by_text = dict(zip(unique_queries, query_vectors))
        document_by_text = dict(zip(unique_documents, document_vectors))
        scores = [
            float(self.np.dot(query_by_text[query], document_by_text[document]))
            for query, document in pairs
        ]
        return scores, {
            "pair_count": len(pairs),
            "unique_query_count": len(unique_queries),
            "unique_document_count": len(unique_documents),
            "batch_size": self.batch_size,
            "model_load_duration_seconds": round(self.load_duration_seconds, 4),
            "inference_duration_seconds": round(duration, 4),
            "mean_pair_latency_ms": round(duration * 1000.0 / max(1, len(pairs)), 4),
            "device": self.device,
            "dtype": "float32_output",
        }


def run(
    *,
    gold_path: Path,
    scorer: Callable[[Sequence[tuple[str, str]]], tuple[list[float], dict[str, Any]]],
    model: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    rows = load_gold_rows(gold_path)
    pairs = [(str(row["query"]), str(row["candidate_text"])) for row in rows]
    scores, telemetry = scorer(pairs)
    comparison, cases = evaluate_reranker_scores(rows, scores)
    improved = comparison["metric_deltas"]["top1_direct"] > 0
    no_quality_regression = comparison["metric_deltas"]["ndcg_at_5"] >= 0
    report = {
        "status": "passed",
        "benchmark_version": "yieldmind-reranker-gold-v1",
        "gold_input": {
            "path": str(gold_path),
            "sha256": _sha256_file(gold_path),
            "row_count": len(rows),
            "case_count": len({row["case_id"] for row in rows}),
            "candidate_scope": "adjudicated hybrid Top-5 candidates only",
        },
        "model": model,
        "inference": telemetry,
        "comparison": comparison,
        "cases": cases,
        "decision": {
            "default_retrieval_changed": False,
            "result": "promising_but_requires_broader_review" if improved and no_quality_regression else "do_not_enable",
            "reason": (
                "Gold Top-1 improved without an nDCG regression, but 30 project-authored cases are insufficient for a production default change."
                if improved and no_quality_regression
                else "The fixed Gold comparison did not improve Top-1 without a ranking-quality regression."
            ),
        },
        "limitations": [
            "The adjudicated set contains 30 project-authored queries and is not a production traffic sample.",
            "Reranking only the saved Top-5 candidates cannot recover a directly relevant chunk that initial retrieval missed.",
            "Candidate text translations were not used; scoring used the original runtime document text.",
        ],
        "real_llm_calls": 0,
        "real_scoring_pair_count": len(rows),
        "real_reranker_model_calls": len(rows) if model.get("scorer_kind") == "cross_encoder" else 0,
        "real_embedding_scoring_pairs": len(rows) if model.get("scorer_kind") == "embedding_cosine" else 0,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "duration_seconds": round(time.time() - started, 4),
    }
    return report


def _write_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"reranker_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--scorer",
        choices=("qwen3-reranker", "qwen3-embedding-cosine"),
        default="qwen3-reranker",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--hf-home",
        default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "hf_cache"),
    )
    args = parser.parse_args()
    if args.max_length < 128 or args.batch_size < 1:
        parser.error("--max-length must be >= 128 and --batch-size must be >= 1")
    os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser().resolve())
    try:
        if args.scorer == "qwen3-reranker":
            scorer = Qwen3Reranker(
                model_id=args.model_id,
                revision=args.model_revision,
                instruction=args.instruction,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                allow_model_download=args.allow_model_download,
            )
            model = {
                "scorer_kind": "cross_encoder",
                "model_id": args.model_id,
                "revision": args.model_revision,
                "instruction": args.instruction,
                "max_length": args.max_length,
                "execution_mode": "local_transformers_real_inference",
                "model_download_allowed": bool(args.allow_model_download),
                "hf_home": os.environ["HF_HOME"],
            }
        else:
            scorer = Qwen3EmbeddingCosineScorer(
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=args.device,
                allow_model_download=args.allow_model_download,
            )
            model = {
                "scorer_kind": "embedding_cosine",
                "model_id": EMBEDDING_MODEL_ID,
                "revision": EMBEDDING_MODEL_REVISION,
                "query_instruction": EMBEDDING_QUERY_INSTRUCTION,
                "document_instruction": "",
                "max_length": args.max_length,
                "execution_mode": "local_sentence_transformer_cosine_rerank_real_inference",
                "model_download_allowed": bool(args.allow_model_download),
                "hf_home": os.environ["HF_HOME"],
            }
        report = run(
            gold_path=Path(args.gold),
            scorer=scorer.score,
            model=model,
        )
        output_path = _write_report(report, Path(args.output_dir))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": str(output_path),
                "comparison": report["comparison"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
