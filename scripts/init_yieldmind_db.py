#!/usr/bin/env python3
"""Initialize the configured YieldMind database."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.checkpointing import initialize_postgres_checkpointer
from yieldmind.database import database_backend, database_location, initialize_database


if __name__ == "__main__":
    target = initialize_database()
    if database_backend(target) == "postgresql":
        initialize_postgres_checkpointer(str(target))
    print(f"YieldMind {database_backend(target)} database initialized: {database_location(target)}")
    if database_backend(target) == "postgresql":
        print("LangGraph PostgreSQL checkpointer initialized.")
