"""Enforce one archive file per task week.

Revision ID: 0018_one_archive_per_week
Revises: 0017_cursor_backed_processing
Create Date: 2026-09-14
"""

from alembic import op

revision = "0018_one_archive_per_week"
down_revision = "0017_cursor_backed_processing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("auto_experiments_promotion_cohort_key", "auto_experiments", type_="unique")
    op.alter_column(
        "auto_experiments",
        "promotion_start_sequence",
        new_column_name="comparison_start_sequence",
    )
    op.alter_column(
        "auto_experiments",
        "promotion_end_sequence",
        new_column_name="comparison_end_sequence",
    )
    op.create_unique_constraint(
        "auto_experiments_comparison_key",
        "auto_experiments",
        ["task_name", "model_id", "comparison_start_sequence", "comparison_end_sequence"],
    )
    op.create_unique_constraint(
        "archive_manifest_task_week_key",
        "archive_manifest",
        ["task_name", "event_date"],
    )


def downgrade() -> None:
    op.drop_constraint("archive_manifest_task_week_key", "archive_manifest", type_="unique")
    op.drop_constraint("auto_experiments_comparison_key", "auto_experiments", type_="unique")
    op.alter_column(
        "auto_experiments",
        "comparison_start_sequence",
        new_column_name="promotion_start_sequence",
    )
    op.alter_column(
        "auto_experiments",
        "comparison_end_sequence",
        new_column_name="promotion_end_sequence",
    )
    op.create_unique_constraint(
        "auto_experiments_promotion_cohort_key",
        "auto_experiments",
        ["task_name", "model_id", "promotion_start_sequence", "promotion_end_sequence"],
    )
