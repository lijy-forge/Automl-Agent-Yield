"""Add layered memory provenance and optimistic session versions."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260918_0005"
down_revision = "20260918_0004"
branch_labels = None
depends_on = None


def _add_missing_columns(table: str, columns: dict[str, sa.Column]) -> None:
    existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)}
    for name, column in columns.items():
        if name not in existing:
            op.add_column(table, column)


def upgrade() -> None:
    _add_missing_columns(
        "yieldmind_sessions",
        {
            "workspace_id": sa.Column("workspace_id", sa.String(length=64), nullable=False, server_default="default"),
            "constraint_version": sa.Column(
                "constraint_version", sa.Integer(), nullable=False, server_default="0"
            ),
            "summary_through_turn_id": sa.Column(
                "summary_through_turn_id", sa.String(length=64), nullable=False, server_default=""
            ),
        },
    )
    _add_missing_columns(
        "yieldmind_turns",
        {
            "constraint_version": sa.Column(
                "constraint_version", sa.Integer(), nullable=False, server_default="0"
            ),
        },
    )
    _add_missing_columns(
        "yieldmind_memories",
        {
            "workspace_id": sa.Column("workspace_id", sa.String(length=64), nullable=False, server_default="default"),
            "validation_status": sa.Column(
                "validation_status", sa.String(length=32), nullable=False, server_default="confirmed"
            ),
            "source_run_id": sa.Column(
                "source_run_id", sa.String(length=64), nullable=False, server_default=""
            ),
            "applicability_json": sa.Column(
                "applicability_json", sa.Text(), nullable=False, server_default="{}"
            ),
        },
    )
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes("yieldmind_memories")}
    if "idx_yieldmind_memories_context" not in indexes:
        op.create_index(
            "idx_yieldmind_memories_context",
            "yieldmind_memories",
            ["workspace_id", "session_id", "scope", "validation_status", "status"],
            unique=False,
        )


def downgrade() -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes("yieldmind_memories")}
    if "idx_yieldmind_memories_context" in indexes:
        op.drop_index("idx_yieldmind_memories_context", table_name="yieldmind_memories")
    for table, columns in (
        ("yieldmind_memories", ("applicability_json", "source_run_id", "validation_status", "workspace_id")),
        ("yieldmind_turns", ("constraint_version",)),
        ("yieldmind_sessions", ("summary_through_turn_id", "constraint_version", "workspace_id")),
    ):
        existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)}
        for column in columns:
            if column in existing:
                op.drop_column(table, column)
