"""Store accepted raw events rather than harness-derived features.

Revision ID: 0013_store_raw_events
Revises: 0012_model_retry_backoff
Create Date: 2026-09-03
"""

from alembic import op

revision = "0013_store_raw_events"
down_revision = "0012_model_retry_backoff"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("benchmark_events", "features", new_column_name="event")


def downgrade() -> None:
    op.alter_column("benchmark_events", "event", new_column_name="features")
