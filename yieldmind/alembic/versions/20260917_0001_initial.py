"""Create the initial YieldMind PostgreSQL schema."""

from __future__ import annotations

from alembic import op

from yieldmind.db_schema import metadata


revision = "20260917_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind=bind)
