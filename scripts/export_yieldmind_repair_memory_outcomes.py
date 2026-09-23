#!/usr/bin/env python
"""Export observed Repair Memory reuse outcomes without running a model."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import YieldMindStore, json_dumps
from yieldmind.repair_memory import RepairMemoryStore


DEFAULT_OUTPUT = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "repair_memory_outcomes"


def export_report(
    store: YieldMindStore,
    *,
    workspace_id: str = "",
    minimum_claimable_observations: int = 30,
) -> dict[str, Any]:
    repair = RepairMemoryStore(store)
    summary = repair.reuse_outcome_summary(
        workspace_id=workspace_id,
        minimum_claimable_observations=minimum_claimable_observations,
    )
    attempts = repair.list_reuse_attempts(workspace_id=workspace_id, limit=1000)
    status = (
        "claimable"
        if summary["claimable"]
        else "preliminary"
        if summary["computable"]
        else "insufficient_evidence"
    )
    return {
        "status": status,
        "report_version": "yieldmind-repair-memory-outcomes-v1",
        "generated_at": time.time(),
        "database_backend": store.backend,
        "database_location": store.location,
        "workspace_id": workspace_id,
        "summary": summary,
        "attempts": attempts,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "python_version": platform.python_version(),
        "limitations": [
            "Only a verified error match that is actually applied to a later revision round enters the denominator.",
            "Task-preflight memories and matches that never reach a revision are excluded because causal reuse was not observed.",
            "The success rate is aggregated by run and applied revision round so multiple memories in one revision do not inflate the denominator.",
            "A computable rate remains preliminary until the configured minimum number of observed reuse episodes is reached.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="", help="Optional SQLite path or PostgreSQL URL.")
    parser.add_argument("--workspace-id", default="")
    parser.add_argument("--minimum-claimable-observations", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    store = YieldMindStore(args.database or None)
    report = export_report(
        store,
        workspace_id=args.workspace_id,
        minimum_claimable_observations=max(1, args.minimum_claimable_observations),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"repair_memory_outcomes_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json_dumps(report) + "\n", encoding="utf-8")
    print(
        json_dumps(
            {
                "status": report["status"],
                "report_path": str(output_path),
                "summary": report["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
