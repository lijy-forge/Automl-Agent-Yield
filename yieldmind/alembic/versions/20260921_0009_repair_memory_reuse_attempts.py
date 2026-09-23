"""Add auditable Repair Memory reuse attempts."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260921_0009"
down_revision = "20260919_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "yieldmind_repair_memory_reuse_attempts" in inspector.get_table_names():
        return
    op.create_table(
        "yieldmind_repair_memory_reuse_attempts",
        sa.Column("reuse_id", sa.String(length=64), primary_key=True),
        sa.Column("memory_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("execution_mode", sa.String(length=64), nullable=False),
        sa.Column("matched_round", sa.Integer(), nullable=False),
        sa.Column("applied_round", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("operation_rcode", sa.Integer()),
        sa.Column("manager_passed", sa.Integer()),
        sa.Column("manager_decision", sa.Text(), nullable=False, server_default=""),
        sa.Column("matched_at", sa.Float(), nullable=False),
        sa.Column("applied_at", sa.Float(), nullable=False),
        sa.Column("completed_at", sa.Float()),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["memory_id"], ["yieldmind_memories.memory_id"]),
        sa.ForeignKeyConstraint(["run_id"], ["yieldmind_runs.run_id"]),
        sa.UniqueConstraint("memory_id", "run_id", "applied_round", name="uq_yieldmind_repair_reuse_attempt"),
    )
    op.create_index(
        "idx_yieldmind_repair_reuse_workspace",
        "yieldmind_repair_memory_reuse_attempts",
        ["workspace_id", "status", "applied_at"],
        unique=False,
    )
    op.create_index(
        "idx_yieldmind_repair_reuse_memory",
        "yieldmind_repair_memory_reuse_attempts",
        ["memory_id", "applied_at"],
        unique=False,
    )


def downgrade() -> None:
    if "yieldmind_repair_memory_reuse_attempts" not in sa.inspect(op.get_bind()).get_table_names():
        return
    op.drop_index("idx_yieldmind_repair_reuse_memory", table_name="yieldmind_repair_memory_reuse_attempts")
    op.drop_index("idx_yieldmind_repair_reuse_workspace", table_name="yieldmind_repair_memory_reuse_attempts")
    op.drop_table("yieldmind_repair_memory_reuse_attempts")
