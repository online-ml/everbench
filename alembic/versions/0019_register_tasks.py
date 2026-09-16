"""Store task names for the dashboard.

Revision ID: 0019_register_tasks
Revises: 0018_one_archive_per_week
"""

import sqlalchemy as sa

from alembic import op

revision = "0019_register_tasks"
down_revision = "0018_one_archive_per_week"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("benchmark_tasks", sa.Column("task_name", sa.String(), primary_key=True))


def downgrade() -> None:
    op.drop_table("benchmark_tasks")
