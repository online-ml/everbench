"""Persist SSE cursors and model failure state.

Revision ID: 0008_stream_cursors_models
Revises: 0007_archive_byte_size
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "0008_stream_cursors_models"
down_revision = "0007_archive_byte_size"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_models", sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("benchmark_models", sa.Column("last_error", sa.Text()))
    op.add_column("benchmark_models", sa.Column("failed_at", sa.DateTime(timezone=True)))
    op.alter_column("benchmark_models", "failure_count", server_default=None)
    op.create_table(
        "stream_cursors",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("stream_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("stream_cursors")
    op.drop_column("benchmark_models", "failed_at")
    op.drop_column("benchmark_models", "last_error")
    op.drop_column("benchmark_models", "failure_count")
