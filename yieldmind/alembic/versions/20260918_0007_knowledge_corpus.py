"""Separate project rules from external literature at retrieval time."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260918_0007"
down_revision = "20260918_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("yieldmind_documents")}
    if "corpus" not in existing:
        op.add_column(
            "yieldmind_documents",
            sa.Column("corpus", sa.String(length=32), nullable=False, server_default="project"),
        )
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes("yieldmind_documents")}
    if "idx_yieldmind_documents_corpus" not in indexes:
        op.create_index(
            "idx_yieldmind_documents_corpus",
            "yieldmind_documents",
            ["index_version", "corpus", "status"],
            unique=False,
        )


def downgrade() -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes("yieldmind_documents")}
    if "idx_yieldmind_documents_corpus" in indexes:
        op.drop_index("idx_yieldmind_documents_corpus", table_name="yieldmind_documents")
    existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("yieldmind_documents")}
    if "corpus" in existing:
        op.drop_column("yieldmind_documents", "corpus")
