"""Add task lease, interruption audit, and recovery lineage."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260918_0004"
down_revision = "20260918_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("yieldmind_tasks")}
    additions = (
        sa.Column("worker_id", sa.String(length=256), nullable=False, server_default=""),
        sa.Column("heartbeat_at", sa.Float(), nullable=True),
        sa.Column("lease_expires_at", sa.Float(), nullable=True),
        sa.Column("interrupted_at", sa.Float(), nullable=True),
        sa.Column("recovery_reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("recovery_requested_by", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("recovery_of_task_id", sa.String(length=64), nullable=False, server_default=""),
    )
    for column in additions:
        if column.name not in columns:
            op.add_column("yieldmind_tasks", column)
    op.create_index(
        "idx_yieldmind_tasks_lease",
        "yieldmind_tasks",
        ["status", "lease_expires_at"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_yieldmind_tasks_recovery",
        "yieldmind_tasks",
        ["recovery_of_task_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("idx_yieldmind_tasks_recovery", table_name="yieldmind_tasks", if_exists=True)
    op.drop_index("idx_yieldmind_tasks_lease", table_name="yieldmind_tasks", if_exists=True)
    for name in (
        "recovery_of_task_id",
        "recovery_requested_by",
        "recovery_reason",
        "interrupted_at",
        "lease_expires_at",
        "heartbeat_at",
        "worker_id",
    ):
        op.drop_column("yieldmind_tasks", name)
