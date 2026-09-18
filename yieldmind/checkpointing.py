"""LangGraph checkpoint backend helpers."""

from __future__ import annotations


def psycopg_connection_string(database_url: str) -> str:
    """Convert a SQLAlchemy PostgreSQL URL into a psycopg connection URL."""
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    if database_url.startswith("postgres+psycopg://"):
        return database_url.replace("postgres+psycopg://", "postgresql://", 1)
    return database_url


def initialize_postgres_checkpointer(database_url: str) -> None:
    """Create or upgrade the tables owned by LangGraph's PostgresSaver."""
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(psycopg_connection_string(database_url)) as checkpointer:
        checkpointer.setup()
