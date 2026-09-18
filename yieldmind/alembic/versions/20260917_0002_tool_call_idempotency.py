"""Add durable idempotency claims for tool calls."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260917_0002"
down_revision = "20260917_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {item["name"] for item in inspector.get_columns("yieldmind_tool_calls")}
    if "idempotency_key" not in columns:
        op.add_column(
            "yieldmind_tool_calls",
            sa.Column("idempotency_key", sa.String(length=128), nullable=False, server_default=""),
        )
    indexes = {item["name"] for item in sa.inspect(bind).get_indexes("yieldmind_tool_calls")}
    if "uq_yieldmind_tool_calls_idempotency" not in indexes:
        op.create_index(
            "uq_yieldmind_tool_calls_idempotency",
            "yieldmind_tool_calls",
            ["idempotency_key"],
            unique=True,
            postgresql_where=sa.text("idempotency_key <> ''"),
        )


def downgrade() -> None:
    op.drop_index("uq_yieldmind_tool_calls_idempotency", table_name="yieldmind_tool_calls")
    op.drop_column("yieldmind_tool_calls", "idempotency_key")
