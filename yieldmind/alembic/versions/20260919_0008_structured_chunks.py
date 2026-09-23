"""Add structured chunk metadata for parent-child retrieval."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260919_0008"
down_revision = "20260918_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    table = "yieldmind_document_chunks"
    existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)}
    additions = {
        "parent_id": sa.Column("parent_id", sa.String(length=64), nullable=False, server_default=""),
        "chunk_role": sa.Column("chunk_role", sa.String(length=32), nullable=False, server_default="content"),
        "token_count": sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        "split_version": sa.Column("split_version", sa.String(length=128), nullable=False, server_default=""),
        "metadata_json": sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
    }
    for name, column in additions.items():
        if name not in existing:
            op.add_column(table, column)
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)}
    if "idx_yieldmind_chunks_parent" not in indexes:
        op.create_index(
            "idx_yieldmind_chunks_parent",
            table,
            ["parent_id", "chunk_index"],
            unique=False,
        )


def downgrade() -> None:
    table = "yieldmind_document_chunks"
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)}
    if "idx_yieldmind_chunks_parent" in indexes:
        op.drop_index("idx_yieldmind_chunks_parent", table_name=table)
    existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)}
    for column in ("metadata_json", "split_version", "token_count", "chunk_role", "parent_id"):
        if column in existing:
            op.drop_column(table, column)
