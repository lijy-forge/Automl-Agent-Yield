"""Add task cancellation audit fields."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260918_0003"
down_revision = "20260917_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("yieldmind_tasks")}
    additions = (
        sa.Column("cancel_requested_at", sa.Float(), nullable=True),
        sa.Column("cancelled_at", sa.Float(), nullable=True),
        sa.Column("cancel_reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("cancel_requested_by", sa.String(length=128), nullable=False, server_default=""),
    )
    for column in additions:
        if column.name not in columns:
            op.add_column("yieldmind_tasks", column)


def downgrade() -> None:
    op.drop_column("yieldmind_tasks", "cancel_requested_by")
    op.drop_column("yieldmind_tasks", "cancel_reason")
    op.drop_column("yieldmind_tasks", "cancelled_at")
    op.drop_column("yieldmind_tasks", "cancel_requested_at")
