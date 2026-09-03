"""Separate model operation errors from intentionally skipped work.

Revision ID: 0014_model_error_and_skip_counts
Revises: 0013_store_raw_events
Create Date: 2026-09-03
"""

import sqlalchemy as sa

from alembic import op

revision = "0014_model_error_and_skip_counts"
down_revision = "0013_store_raw_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("benchmark_models", "prediction_errors", new_column_name="skipped_predictions")
    op.alter_column("benchmark_models", "label_errors", new_column_name="skipped_labels")
    op.add_column("benchmark_models", sa.Column("error_count", sa.BigInteger(), nullable=False, server_default="0"))
    op.execute("UPDATE benchmark_models SET error_count = failure_count")
    op.alter_column("benchmark_models", "error_count", server_default=None)


def downgrade() -> None:
    op.drop_column("benchmark_models", "error_count")
    op.alter_column("benchmark_models", "skipped_labels", new_column_name="label_errors")
    op.alter_column("benchmark_models", "skipped_predictions", new_column_name="prediction_errors")
