#!/usr/bin/env python3
"""Run the YieldMind workflow layer once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.workflow import WorkflowRequest, run_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YieldMind workflow.")
    parser.add_argument("--prompt", default="Run a deterministic yield baseline evaluation.")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    result = run_workflow(
        WorkflowRequest(
            prompt=args.prompt,
            data_path=args.data_path,
            n_samples=args.n_samples,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
