"""Add cursor-backed event, label, and horizon work queues.

Revision ID: 0017_cursor_backed_processing
Revises: 0016_auto_experiments
Create Date: 2026-09-06
"""

import sqlalchemy as sa

from alembic import op

revision = "0017_cursor_backed_processing"
down_revision = "0016_auto_experiments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_ready_labels",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("sequence", sa.BigInteger(), sa.Identity(), nullable=False, unique=True),
        sa.ForeignKeyConstraint(
            ["task_name", "event_id"],
            ["benchmark_events.task_name", "benchmark_events.event_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_name", "event_id"],
            ["benchmark_labels.task_name", "benchmark_labels.event_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "benchmark_ready_labels_task_sequence_idx",
        "benchmark_ready_labels",
        ["task_name", "sequence"],
    )
    op.execute(
        """INSERT INTO benchmark_ready_labels (task_name, event_id)
           SELECT label.task_name, label.event_id
             FROM benchmark_labels AS label
             JOIN benchmark_events AS event USING (task_name, event_id)
            ORDER BY GREATEST(label.inserted_at, event.inserted_at), event.sequence"""
    )

    op.create_table(
        "benchmark_negative_label_schedule",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_name", "event_id"],
            ["benchmark_events.task_name", "benchmark_events.event_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "benchmark_negative_label_schedule_due_idx",
        "benchmark_negative_label_schedule",
        ["task_name", "due_at"],
    )

    op.add_column(
        "benchmark_models",
        sa.Column("prediction_cursor_sequence", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "benchmark_models",
        sa.Column("label_cursor_sequence", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.execute(
        """UPDATE benchmark_models AS model
              SET prediction_cursor_sequence = COALESCE(
                    (SELECT MIN(event.sequence) - 1
                       FROM benchmark_events AS event
                       LEFT JOIN benchmark_model_events AS state
                         ON state.task_name = event.task_name
                        AND state.event_id = event.event_id
                        AND state.model_id = model.model_id
                      WHERE event.task_name = model.task_name
                        AND event.sequence >= model.start_sequence
                        AND state.event_id IS NULL),
                    (SELECT COALESCE(MAX(event.sequence), model.start_sequence - 1)
                       FROM benchmark_events AS event
                      WHERE event.task_name = model.task_name)
                  ),
                  label_cursor_sequence = COALESCE(
                    (SELECT MIN(ready.sequence) - 1
                       FROM benchmark_ready_labels AS ready
                       JOIN benchmark_events AS event USING (task_name, event_id)
                       LEFT JOIN benchmark_model_events AS state
                         ON state.task_name = ready.task_name
                        AND state.event_id = ready.event_id
                        AND state.model_id = model.model_id
                      WHERE ready.task_name = model.task_name
                        AND event.sequence >= model.start_sequence
                        AND (state.event_id IS NULL OR state.trained_at IS NULL
                             OR (state.prediction_status = 'predicted' AND state.evaluated_at IS NULL))),
                    (SELECT COALESCE(MAX(ready.sequence), 0)
                       FROM benchmark_ready_labels AS ready
                      WHERE ready.task_name = model.task_name)
                  )"""
    )
    op.alter_column("benchmark_models", "prediction_cursor_sequence", server_default=None)
    op.alter_column("benchmark_models", "label_cursor_sequence", server_default=None)
    op.add_column("model_snapshots", sa.Column("checkpoint_ready_sequence", sa.BigInteger()))


def downgrade() -> None:
    op.drop_column("model_snapshots", "checkpoint_ready_sequence")
    op.drop_column("benchmark_models", "label_cursor_sequence")
    op.drop_column("benchmark_models", "prediction_cursor_sequence")
    op.drop_index("benchmark_negative_label_schedule_due_idx", table_name="benchmark_negative_label_schedule")
    op.drop_table("benchmark_negative_label_schedule")
    op.drop_index("benchmark_ready_labels_task_sequence_idx", table_name="benchmark_ready_labels")
    op.drop_table("benchmark_ready_labels")
