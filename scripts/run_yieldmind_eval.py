#!/usr/bin/env python3
"""Run the deterministic offline YieldMind evaluation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.evaluation import report_to_text, run_offline_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YieldMind offline evaluation.")
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    report = run_offline_evaluation(python_executable=args.python_executable)
    print(report_to_text(report))
    return 0 if report["summary"]["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
