#!/usr/bin/env python3
"""Start the YieldMind FastAPI service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="Start YieldMind API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("yieldmind.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
