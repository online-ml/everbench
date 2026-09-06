"""Add autonomous research experiment ledger.

Revision ID: 0016_auto_experiments
Revises: 0015_model_event_state
Create Date: 2026-09-06
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0016_auto_experiments"
down_revision = "0015_model_event_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "auto_experiments",
        sa.Column("experiment_id", sa.String(), primary_key=True),
        sa.Column("task_name", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("parent_generation", sa.Integer(), nullable=False),
        sa.Column("researcher", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("hypothesis", sa.Text()),
        sa.Column("proposal", postgresql.JSONB()),
        sa.Column("research_summary", postgresql.JSONB()),
        sa.Column("evaluation", postgresql.JSONB()),
        sa.Column("promotion_start_sequence", sa.BigInteger(), nullable=False),
        sa.Column("promotion_end_sequence", sa.BigInteger(), nullable=False),
        sa.Column("champion_artifact_id", sa.String(), nullable=False),
        sa.Column("candidate_artifact_id", sa.String()),
        sa.Column("error", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('running', 'rejected', 'promoted', 'failed')", name="auto_experiment_status"
        ),
        sa.ForeignKeyConstraint(
            ["task_name", "model_id"],
            ["benchmark_models.task_name", "benchmark_models.model_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["champion_artifact_id"], ["model_artifacts.artifact_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["candidate_artifact_id"], ["model_artifacts.artifact_id"], ondelete="RESTRICT"),
        sa.UniqueConstraint(
            "task_name",
            "model_id",
            "promotion_start_sequence",
            "promotion_end_sequence",
            name="auto_experiments_promotion_cohort_key",
        ),
    )
    op.create_index(
        "auto_experiments_latest_idx", "auto_experiments", ["task_name", "model_id", "started_at"]
    )


def downgrade() -> None:
    op.drop_index("auto_experiments_latest_idx", table_name="auto_experiments")
    op.drop_table("auto_experiments")
