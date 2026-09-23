#!/usr/bin/env python
"""Run paired baseline-vs-recovery controlled-failure evaluation."""

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
from yieldmind.recovery_eval import evaluate_recovery_suite, validate_recovery_eval_payload


DEFAULT_CASES = PROJECT_ROOT / "evals" / "yieldmind_recovery_cases_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "agent_workspace" / "yieldmind" / "recovery_evals"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        suite = validate_recovery_eval_payload(json.loads(args.cases.read_text(encoding="utf-8")))
        report = evaluate_recovery_suite(suite)
    except Exception as exc:
        print(json_dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"recovery_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json_dumps(report) + "\n", encoding="utf-8")
    print(
        json_dumps(
            {
                "status": report["status"],
                "report_path": str(output_path),
                "case_count": report["case_count"],
                "metrics": report["metrics"],
            }
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
