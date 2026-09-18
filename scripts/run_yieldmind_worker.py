#!/usr/bin/env python3
"""Run a YieldMind Celery worker for the dedicated queue."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.task_queue import celery_app


if __name__ == "__main__":
    celery_app.worker_main(["worker", "--loglevel=INFO", "--pool=solo", "--queues=yieldmind"])
