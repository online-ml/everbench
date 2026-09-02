"""Add missed-prediction receipts, checkpoint watermarks, and query indexes.

Revision ID: 0006_runtime_hardening
Revises: 0005_model_owner
Create Date: 2026-09-02
"""

import sqlalchemy as sa

from alembic import op

revision = "0006_runtime_hardening"
down_revision = "0005_model_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_prediction_skips",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("skipped_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )
    op.add_column("model_snapshots", sa.Column("checkpoint_label_available_at", sa.DateTime(timezone=True)))
    op.add_column("model_snapshots", sa.Column("checkpoint_event_sequence", sa.BigInteger()))

    # Existing primary keys start with event ID, whereas the live learner
    # usually filters receipt tables by (task, model).
    op.create_index("benchmark_events_inserted_idx", "benchmark_events", ["task_name", "inserted_at"])
    op.create_index("benchmark_labels_available_idx", "benchmark_labels", ["task_name", "available_at"])
    op.create_index("benchmark_predictions_model_idx", "benchmark_predictions", ["task_name", "model_id", "event_id"])
    op.create_index(
        "benchmark_prediction_skips_model_idx", "benchmark_prediction_skips", ["task_name", "model_id", "event_id"]
    )
    op.create_index("benchmark_trainings_model_idx", "benchmark_trainings", ["task_name", "model_id", "event_id"])
    op.create_index(
        "benchmark_metric_updates_model_idx", "benchmark_metric_updates", ["task_name", "model_id", "event_id"]
    )


def downgrade() -> None:
    for name in (
        "benchmark_metric_updates_model_idx",
        "benchmark_trainings_model_idx",
        "benchmark_prediction_skips_model_idx",
        "benchmark_predictions_model_idx",
        "benchmark_labels_available_idx",
        "benchmark_events_inserted_idx",
    ):
        op.drop_index(name)
    op.drop_column("model_snapshots", "checkpoint_event_sequence")
    op.drop_column("model_snapshots", "checkpoint_label_available_at")
    op.drop_table("benchmark_prediction_skips")
