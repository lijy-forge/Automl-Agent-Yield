#!/usr/bin/env python3
"""Run the original yield multi-agent phases through LangGraph StateGraph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.domain_workflow import DomainWorkflowRequest, run_domain_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the live YieldMind domain StateGraph.")
    parser.add_argument("--prompt", default="Build an AutoML model for high-solid-content slurry yield-stress prediction.")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--test-path", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--llm", default="openai")
    parser.add_argument("--n-revise", type=int, default=1)
    parser.add_argument("--operation-attempts", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--synthetic-data", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--synthetic-n", type=int, default=800)
    parser.add_argument("--external-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-search-results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument(
        "--execution-mode",
        choices=("free_search", "plugin", "llm_freeform"),
        default="free_search",
    )
    parser.add_argument("--operation-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--allow-live-llm",
        action="store_true",
        help="Required acknowledgement that CandidateAgent/ModelAgent may call the configured model API.",
    )
    args = parser.parse_args()

    request = DomainWorkflowRequest(
        prompt=args.prompt,
        data_path=args.data_path,
        test_path=args.test_path,
        run_dir=args.run_dir,
        llm=args.llm,
        n_revise=args.n_revise,
        operation_attempts=args.operation_attempts,
        random_state=args.random_state,
        synthetic_data=args.synthetic_data,
        synthetic_n=args.synthetic_n,
        external_search=args.external_search,
        require_search_results=args.require_search_results,
        query=args.query,
        execution_mode=args.execution_mode,
        operation_timeout_seconds=args.operation_timeout_seconds,
        allow_live_llm=args.allow_live_llm,
    )
    result = run_domain_workflow(request)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "passed" else 2 if result.get("status") == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
