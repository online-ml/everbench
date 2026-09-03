"""Consolidate per-model event receipts and enforce model ownership.

Revision ID: 0015_model_event_state
Revises: 0014_model_error_and_skip_counts
Create Date: 2026-09-03
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0015_model_event_state"
down_revision = "0014_model_error_and_skip_counts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "benchmark_labels",
        sa.Column("inserted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("benchmark_labels_inserted_idx", "benchmark_labels", ["task_name", "inserted_at"])

    op.create_table(
        "benchmark_model_events",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("prediction", postgresql.JSONB()),
        sa.Column("prediction_status", sa.String(), nullable=False),
        sa.Column("prediction_reason", sa.Text()),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("evaluated_at", sa.DateTime(timezone=True)),
        sa.Column("trained_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("prediction_status IN ('predicted', 'skipped')", name="model_event_prediction_status"),
        sa.CheckConstraint(
            "(prediction_status = 'predicted' AND prediction IS NOT NULL AND prediction_reason IS NULL) "
            "OR (prediction_status = 'skipped' AND prediction IS NULL AND prediction_reason IS NOT NULL)",
            name="model_event_prediction_payload",
        ),
        sa.ForeignKeyConstraint(
            ["task_name", "event_id"],
            ["benchmark_events.task_name", "benchmark_events.event_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_name", "model_id"],
            ["benchmark_models.task_name", "benchmark_models.model_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "benchmark_model_events_model_idx",
        "benchmark_model_events",
        ["task_name", "model_id", "event_id"],
    )

    op.execute(
        """INSERT INTO benchmark_model_events
             (task_name, event_id, model_id, prediction, prediction_status, predicted_at, evaluated_at, trained_at)
           SELECT prediction.task_name,
                  prediction.event_id,
                  prediction.model_id,
                  prediction.prediction,
                  'predicted',
                  prediction.predicted_at,
                  metric_update.updated_at,
                  training.trained_at
             FROM benchmark_predictions AS prediction
             JOIN benchmark_models AS model USING (task_name, model_id)
             LEFT JOIN benchmark_metric_updates AS metric_update USING (task_name, event_id, model_id)
             LEFT JOIN benchmark_trainings AS training USING (task_name, event_id, model_id)"""
    )
    op.execute(
        """INSERT INTO benchmark_model_events
             (task_name, event_id, model_id, prediction_status, prediction_reason, predicted_at, trained_at)
           SELECT skip.task_name,
                  skip.event_id,
                  skip.model_id,
                  'skipped',
                  skip.reason,
                  skip.skipped_at,
                  training.trained_at
             FROM benchmark_prediction_skips AS skip
             JOIN benchmark_models AS model USING (task_name, model_id)
             LEFT JOIN benchmark_trainings AS training USING (task_name, event_id, model_id)
           ON CONFLICT (task_name, event_id, model_id) DO NOTHING"""
    )

    for table in (
        "benchmark_metric_updates",
        "benchmark_trainings",
        "benchmark_prediction_skips",
        "benchmark_predictions",
    ):
        op.drop_table(table)

    for table in ("benchmark_metric_state", "model_snapshots"):
        op.execute(
            f"""DELETE FROM {table} AS state
                  WHERE NOT EXISTS (
                    SELECT 1 FROM benchmark_models AS model
                     WHERE model.task_name = state.task_name AND model.model_id = state.model_id
                  )"""
        )
        op.create_foreign_key(
            f"{table}_model_fkey",
            table,
            "benchmark_models",
            ["task_name", "model_id"],
            ["task_name", "model_id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    op.drop_constraint("model_snapshots_model_fkey", "model_snapshots", type_="foreignkey")
    op.drop_constraint("benchmark_metric_state_model_fkey", "benchmark_metric_state", type_="foreignkey")
    op.create_table(
        "benchmark_predictions",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("prediction", postgresql.JSONB(), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )
    op.create_table(
        "benchmark_prediction_skips",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("skipped_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )
    op.create_table(
        "benchmark_trainings",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )
    op.create_table(
        "benchmark_metric_updates",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )
    op.create_index("benchmark_predictions_model_idx", "benchmark_predictions", ["task_name", "model_id", "event_id"])
    op.create_index(
        "benchmark_prediction_skips_model_idx",
        "benchmark_prediction_skips",
        ["task_name", "model_id", "event_id"],
    )
    op.create_index("benchmark_trainings_model_idx", "benchmark_trainings", ["task_name", "model_id", "event_id"])
    op.create_index(
        "benchmark_metric_updates_model_idx",
        "benchmark_metric_updates",
        ["task_name", "model_id", "event_id"],
    )
    op.execute(
        """INSERT INTO benchmark_predictions
             (task_name, event_id, model_id, prediction, predicted_at)
           SELECT task_name, event_id, model_id, prediction, predicted_at
             FROM benchmark_model_events WHERE prediction_status = 'predicted'"""
    )
    op.execute(
        """INSERT INTO benchmark_prediction_skips
             (task_name, event_id, model_id, reason, skipped_at)
           SELECT task_name, event_id, model_id, prediction_reason, predicted_at
             FROM benchmark_model_events WHERE prediction_status = 'skipped'"""
    )
    op.execute(
        """INSERT INTO benchmark_trainings (task_name, event_id, model_id, trained_at)
           SELECT task_name, event_id, model_id, trained_at
             FROM benchmark_model_events WHERE trained_at IS NOT NULL"""
    )
    op.execute(
        """INSERT INTO benchmark_metric_updates (task_name, event_id, model_id, updated_at)
           SELECT task_name, event_id, model_id, evaluated_at
             FROM benchmark_model_events WHERE evaluated_at IS NOT NULL"""
    )
    op.drop_table("benchmark_model_events")
    op.drop_index("benchmark_labels_inserted_idx", table_name="benchmark_labels")
    op.drop_column("benchmark_labels", "inserted_at")
