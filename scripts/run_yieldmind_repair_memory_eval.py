#!/usr/bin/env python
"""Run the deterministic Repair Memory retrieval and safety benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.database import json_dumps
from yieldmind.repair_memory_eval import (
    evaluate_repair_memory_suite,
    validate_repair_memory_eval_payload,
)


DEFAULT_CASES = PROJECT_ROOT / "evals" / "yieldmind_repair_memory_cases_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "repair_memory_evals"


def run(*, cases_path: Path, output_dir: Path) -> tuple[dict[str, object], Path]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    suite = validate_repair_memory_eval_payload(payload)
    report = evaluate_repair_memory_suite(suite)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"repair_memory_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json_dumps(report) + "\n", encoding="utf-8")
    return report, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-threshold-failures",
        action="store_true",
        help="Write diagnostics but exit successfully when one or more quality thresholds fail.",
    )
    args = parser.parse_args()
    try:
        report, output_path = run(cases_path=args.cases, output_dir=args.output_dir)
    except Exception as exc:
        print(json_dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 1

    print(
        json_dumps(
            {
                "status": report["status"],
                "report_path": str(output_path),
                "metrics": report["metrics"],
                "repair_success": report["repair_success"],
                "threshold_checks": report["threshold_checks"],
            }
        )
    )
    if report["status"] != "passed" and not args.allow_threshold_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
