"""Allow labels to arrive before their corresponding event.

Revision ID: 0003_label_inbox
Revises: 0002_artifacts_and_archives
Create Date: 2026-09-02
"""

from alembic import op

revision = "0003_label_inbox"
down_revision = "0002_artifacts_and_archives"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("benchmark_labels_task_name_event_id_fkey", "benchmark_labels", type_="foreignkey")


def downgrade() -> None:
    op.create_foreign_key(
        "benchmark_labels_task_name_event_id_fkey",
        "benchmark_labels",
        "benchmark_events",
        ["task_name", "event_id"],
        ["task_name", "event_id"],
    )
