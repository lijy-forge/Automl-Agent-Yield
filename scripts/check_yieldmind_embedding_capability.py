#!/usr/bin/env python3
"""Audit local embedding readiness without network access or model downloads."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import json_dumps


CANDIDATES = [
    {
        "name": "Qwen3 Embedding 0.6B",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "native_dimensions": 1024,
        "minimum_transformers": "4.51.0",
        "minimum_sentence_transformers": "2.7.0",
        "recommended_backend": "sentence_transformers",
    },
    {
        "name": "BGE-M3",
        "model_id": "BAAI/bge-m3",
        "native_dimensions": 1024,
        "recommended_backend": "FlagEmbedding",
    },
]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _meets(installed: str, minimum: str) -> bool:
    return installed != "not-installed" and Version(installed) >= Version(minimum)


def _cache_path(model_id: str) -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return hub / f"models--{model_id.replace('/', '--')}"


def _memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def build_report() -> dict[str, Any]:
    transformers_version = _package_version("transformers")
    sentence_transformers_version = _package_version("sentence-transformers")
    torch_version = _package_version("torch")
    try:
        import torch

        accelerators = {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except Exception as exc:
        accelerators = {"cuda_available": False, "mps_available": False, "error": f"{type(exc).__name__}: {exc}"}

    candidates = []
    for candidate in CANDIDATES:
        reasons = []
        cached = _cache_path(candidate["model_id"]).exists()
        if not cached:
            reasons.append("model_not_cached")
        minimum_transformers = candidate.get("minimum_transformers")
        if minimum_transformers and not _meets(transformers_version, minimum_transformers):
            reasons.append(f"transformers_requires_at_least_{minimum_transformers}")
        minimum_st = candidate.get("minimum_sentence_transformers")
        if minimum_st and not _meets(sentence_transformers_version, minimum_st):
            reasons.append(f"sentence_transformers_requires_at_least_{minimum_st}")
        if candidate["recommended_backend"] == "FlagEmbedding" and importlib.util.find_spec("FlagEmbedding") is None:
            reasons.append("FlagEmbedding_not_installed")
        if not accelerators["cuda_available"] and not accelerators["mps_available"]:
            reasons.append("cpu_only_benchmark_required")
        candidates.append(
            candidate
            | {
                "cache_path": str(_cache_path(candidate["model_id"])),
                "cached": cached,
                "ready_without_changes": not reasons,
                "blocking_or_cost_factors": reasons,
            }
        )

    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "status": "passed",
        "mode": "offline_capability_audit",
        "network_calls": 0,
        "model_downloads": 0,
        "model_inference_calls": 0,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch_version,
            "transformers": transformers_version,
            "sentence_transformers": sentence_transformers_version,
            "FlagEmbedding": _package_version("FlagEmbedding"),
            "memory_bytes": _memory_bytes(),
            "disk_free_bytes": disk.free,
            **accelerators,
        },
        "candidates": candidates,
        "decision": {
            "production_default_changed": False,
            "current_default": "local_hashing_v1_for_offline_regression_only",
            "next_required_step": (
                "Create an isolated compatible environment, pin an immutable model revision, then run the same "
                "30-query benchmark and record quality, peak memory, index time, and query latency."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "embedding_capability"),
    )
    args = parser.parse_args()
    report = build_report()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"embedding_capability_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json_dumps(report) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "report_path": str(output_path),
        "environment": report["environment"],
        "candidates": [
            {
                "model_id": item["model_id"],
                "ready_without_changes": item["ready_without_changes"],
                "blocking_or_cost_factors": item["blocking_or_cost_factors"],
            }
            for item in report["candidates"]
        ],
    }
    print(json_dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
