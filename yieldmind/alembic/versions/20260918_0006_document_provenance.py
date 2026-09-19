"""Add document provenance and chunk location metadata."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260918_0006"
down_revision = "20260918_0005"
branch_labels = None
depends_on = None


def _add_missing_columns(table: str, columns: dict[str, sa.Column]) -> None:
    existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)}
    for name, column in columns.items():
        if name not in existing:
            op.add_column(table, column)


def upgrade() -> None:
    _add_missing_columns(
        "yieldmind_documents",
        {
            "source_url": sa.Column("source_url", sa.Text(), nullable=False, server_default=""),
            "doi": sa.Column("doi", sa.Text(), nullable=False, server_default=""),
            "license": sa.Column("license", sa.Text(), nullable=False, server_default=""),
            "metadata_json": sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        },
    )
    _add_missing_columns(
        "yieldmind_document_chunks",
        {
            "page_start": sa.Column("page_start", sa.Integer(), nullable=True),
            "page_end": sa.Column("page_end", sa.Integer(), nullable=True),
        },
    )


def downgrade() -> None:
    for table, columns in (
        ("yieldmind_document_chunks", ("page_end", "page_start")),
        ("yieldmind_documents", ("metadata_json", "license", "doi", "source_url")),
    ):
        existing = {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)}
        for column in columns:
            if column in existing:
                op.drop_column(table, column)
