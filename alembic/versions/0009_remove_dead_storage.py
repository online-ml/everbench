"""Remove unused event and model state columns.

Revision ID: 0009_remove_dead_storage
Revises: 0008_stream_cursors_models
Create Date: 2026-09-03
"""

import sqlalchemy as sa

from alembic import op

revision = "0009_remove_dead_storage"
down_revision = "0008_stream_cursors_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("benchmark_model_state")
    op.drop_column("benchmark_events", "archived_at")
    op.drop_column("benchmark_events", "payload")
    op.drop_column("benchmark_models", "config")
    op.drop_column("benchmark_models", "kind")


def downgrade() -> None:
    op.add_column("benchmark_models", sa.Column("kind", sa.String(), nullable=False, server_default="pickle"))
    op.add_column("benchmark_models", sa.Column("config", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
    op.alter_column("benchmark_models", "kind", server_default=None)
    op.alter_column("benchmark_models", "config", server_default=None)
    op.add_column("benchmark_events", sa.Column("payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
    op.add_column("benchmark_events", sa.Column("archived_at", sa.DateTime(timezone=True)))
    op.alter_column("benchmark_events", "payload", server_default=None)
    op.create_table(
        "benchmark_model_state",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
