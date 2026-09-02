"""Generalize targets and predictions; persist River metric checkpoints.

Revision ID: 0004_task_metrics
Revises: 0003_label_inbox
Create Date: 2026-09-02
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004_task_metrics"
down_revision = "0003_label_inbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The first schema only supported binary labels and probability scores.
    # JSONB preserves those existing values while allowing every task family.
    op.execute("ALTER TABLE benchmark_labels DROP CONSTRAINT IF EXISTS benchmark_labels_y_check")
    op.execute("ALTER TABLE benchmark_labels ALTER COLUMN y TYPE jsonb USING to_jsonb(y)")
    op.alter_column("benchmark_predictions", "score", new_column_name="prediction")
    op.execute("ALTER TABLE benchmark_predictions ALTER COLUMN prediction TYPE jsonb USING to_jsonb(prediction)")

    op.create_table(
        "benchmark_metric_state",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("definition", postgresql.JSONB(), nullable=False),
        sa.Column("state", sa.LargeBinary(), nullable=False),
        sa.Column("predictions", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("observations", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("values", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_table(
        "benchmark_metric_updates",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )


def downgrade() -> None:
    op.drop_table("benchmark_metric_updates")
    op.drop_table("benchmark_metric_state")
    op.execute(
        "ALTER TABLE benchmark_predictions ALTER COLUMN prediction TYPE double precision USING (prediction #>> '{}')::double precision"
    )
    op.alter_column("benchmark_predictions", "prediction", new_column_name="score")
    op.execute("ALTER TABLE benchmark_labels ALTER COLUMN y TYPE smallint USING (y #>> '{}')::smallint")
    op.create_check_constraint("benchmark_labels_y_check", "benchmark_labels", "y IN (0, 1)")
