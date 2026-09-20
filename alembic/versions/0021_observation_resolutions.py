"""Generalize label deadlines and retain unavailable observations in archives.

Revision ID: 0021_observation_resolutions
Revises: 0020_task_live_counts
"""

import sqlalchemy as sa

from alembic import op

revision = "0021_observation_resolutions"
down_revision = "0020_task_live_counts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "benchmark_model_events", sa.Column("training_skipped", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    op.execute(
        "UPDATE benchmark_model_events SET training_skipped = true WHERE prediction_reason = 'model-disabled' AND trained_at IS NOT NULL"
    )
    op.rename_table("benchmark_negative_label_schedule", "benchmark_label_schedule")
    op.execute("ALTER INDEX benchmark_negative_label_schedule_due_idx RENAME TO benchmark_label_schedule_due_idx")
    op.add_column("benchmark_label_schedule", sa.Column("target_at", sa.DateTime(timezone=True)))
    op.add_column("benchmark_label_schedule", sa.Column("entity_key", sa.String()))
    op.execute("UPDATE benchmark_label_schedule SET target_at = due_at, due_at = due_at + INTERVAL '1 minute'")
    op.alter_column("benchmark_label_schedule", "target_at", nullable=False)
    op.create_index(
        "benchmark_label_schedule_entity_idx", "benchmark_label_schedule", ["task_name", "entity_key", "target_at"]
    )
    op.add_column(
        "benchmark_ready_labels", sa.Column("has_target", sa.Boolean(), nullable=False, server_default=sa.true())
    )
    op.add_column("archive_manifest", sa.Column("label_count", sa.Integer()))
    # Existing archives and ready labels all contain known targets. The nullable
    # manifest count preserves compatibility with their immutable Parquet bytes.
    op.execute("UPDATE archive_manifest SET label_count = row_count")
    op.execute("""CREATE OR REPLACE FUNCTION benchmark_task_live_counts() RETURNS trigger LANGUAGE plpgsql AS $$
       BEGIN
           IF TG_TABLE_NAME = 'benchmark_events' AND TG_OP = 'INSERT' THEN
               INSERT INTO benchmark_tasks (task_name, live_events)
               SELECT task_name, COUNT(*) FROM new_rows GROUP BY task_name
               ON CONFLICT (task_name) DO UPDATE SET live_events = benchmark_tasks.live_events + EXCLUDED.live_events;
           ELSIF TG_TABLE_NAME = 'benchmark_events' AND TG_OP = 'DELETE' THEN
               UPDATE benchmark_tasks AS task SET live_events = task.live_events - deleted.row_count
               FROM (SELECT task_name, COUNT(*) AS row_count FROM old_rows GROUP BY task_name) AS deleted
               WHERE task.task_name = deleted.task_name;
           ELSIF TG_TABLE_NAME = 'benchmark_ready_labels' AND TG_OP = 'INSERT' THEN
               INSERT INTO benchmark_tasks (task_name, live_labels)
               SELECT task_name, COUNT(*) FILTER (WHERE has_target) FROM new_rows GROUP BY task_name
               ON CONFLICT (task_name) DO UPDATE SET live_labels = benchmark_tasks.live_labels + EXCLUDED.live_labels;
           ELSIF TG_TABLE_NAME = 'benchmark_ready_labels' AND TG_OP = 'DELETE' THEN
               UPDATE benchmark_tasks AS task SET live_labels = task.live_labels - deleted.row_count
               FROM (SELECT task_name, COUNT(*) FILTER (WHERE has_target) AS row_count FROM old_rows GROUP BY task_name) AS deleted
               WHERE task.task_name = deleted.task_name;
           END IF;
           RETURN NULL;
       END $$""")


def downgrade() -> None:
    raise RuntimeError(
        "unavailable observations cannot be represented by the previous schema; restore a backup to downgrade"
    )
