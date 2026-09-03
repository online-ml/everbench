"""Initial shared everbench schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-02
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_events",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("sequence", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("inserted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("task_name", "sequence"),
    )
    op.create_index("benchmark_events_time_idx", "benchmark_events", ["task_name", "event_time"])
    op.create_table(
        "benchmark_labels",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("y", sa.SmallInteger(), sa.CheckConstraint("y IN (0, 1)"), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )
    for table in ("benchmark_predictions", "benchmark_trainings"):
        columns = [
            sa.Column("task_name", sa.String(), primary_key=True),
            sa.Column("event_id", sa.String(), primary_key=True),
            sa.Column("model_id", sa.String(), primary_key=True),
        ]
        if table == "benchmark_predictions":
            columns.append(sa.Column("score", sa.Float(), nullable=False))
            columns.append(
                sa.Column("predicted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False)
            )
        else:
            columns.append(
                sa.Column("trained_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False)
            )
        columns.append(
            sa.ForeignKeyConstraint(
                ["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]
            )
        )
        op.create_table(table, *columns)
    op.create_table(
        "benchmark_model_state",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("state", postgresql.JSONB(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "benchmark_models",
        sa.Column("task_name", sa.String(), primary_key=True),
        sa.Column("model_id", sa.String(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("start_sequence", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("benchmark_models_active_idx", "benchmark_models", ["task_name", "active"])
    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.String(), primary_key=True),
        sa.Column("task_name", sa.String()),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("detail", sa.Text()),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "worker_heartbeats",
        "benchmark_models",
        "benchmark_model_state",
        "benchmark_trainings",
        "benchmark_predictions",
        "benchmark_labels",
        "benchmark_events",
    ):
        op.drop_table(table)
