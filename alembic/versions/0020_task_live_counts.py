"""Maintain exact live task counts as event and ready-label batches change.

Revision ID: 0020_task_live_counts
Revises: 0019_register_tasks
"""

import sqlalchemy as sa

from alembic import op

revision = "0020_task_live_counts"
down_revision = "0019_register_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("benchmark_tasks", sa.Column("live_events", sa.BigInteger(), server_default="0", nullable=False))
    op.add_column("benchmark_tasks", sa.Column("live_labels", sa.BigInteger(), server_default="0", nullable=False))

    # Hold writers until the snapshot and triggers commit together, so no batch
    # can be counted twice or missed while the existing rows are backfilled.
    op.execute("LOCK TABLE benchmark_events, benchmark_ready_labels IN SHARE ROW EXCLUSIVE MODE")
    op.execute(
        """UPDATE benchmark_tasks AS task
              SET live_events = (SELECT COUNT(*) FROM benchmark_events WHERE task_name = task.task_name),
                  live_labels = (SELECT COUNT(*) FROM benchmark_ready_labels WHERE task_name = task.task_name)"""
    )
    op.execute(
        """CREATE FUNCTION benchmark_task_live_counts() RETURNS trigger LANGUAGE plpgsql AS $$
           BEGIN
               IF TG_TABLE_NAME = 'benchmark_events' AND TG_OP = 'INSERT' THEN
                   INSERT INTO benchmark_tasks (task_name, live_events)
                   SELECT task_name, COUNT(*) FROM new_rows GROUP BY task_name
                   ON CONFLICT (task_name) DO UPDATE
                       SET live_events = benchmark_tasks.live_events + EXCLUDED.live_events;
               ELSIF TG_TABLE_NAME = 'benchmark_events' AND TG_OP = 'DELETE' THEN
                   UPDATE benchmark_tasks AS task
                      SET live_events = task.live_events - deleted.row_count
                     FROM (SELECT task_name, COUNT(*) AS row_count FROM old_rows GROUP BY task_name) AS deleted
                    WHERE task.task_name = deleted.task_name;
               ELSIF TG_TABLE_NAME = 'benchmark_ready_labels' AND TG_OP = 'INSERT' THEN
                   INSERT INTO benchmark_tasks (task_name, live_labels)
                   SELECT task_name, COUNT(*) FROM new_rows GROUP BY task_name
                   ON CONFLICT (task_name) DO UPDATE
                       SET live_labels = benchmark_tasks.live_labels + EXCLUDED.live_labels;
               ELSIF TG_TABLE_NAME = 'benchmark_ready_labels' AND TG_OP = 'DELETE' THEN
                   UPDATE benchmark_tasks AS task
                      SET live_labels = task.live_labels - deleted.row_count
                     FROM (SELECT task_name, COUNT(*) AS row_count FROM old_rows GROUP BY task_name) AS deleted
                    WHERE task.task_name = deleted.task_name;
               END IF;
               RETURN NULL;
           END $$"""
    )
    for table in ("benchmark_events", "benchmark_ready_labels"):
        op.execute(
            f"""CREATE TRIGGER {table}_count_insert AFTER INSERT ON {table}
                REFERENCING NEW TABLE AS new_rows FOR EACH STATEMENT
                EXECUTE FUNCTION benchmark_task_live_counts()"""
        )
        op.execute(
            f"""CREATE TRIGGER {table}_count_delete AFTER DELETE ON {table}
                REFERENCING OLD TABLE AS old_rows FOR EACH STATEMENT
                EXECUTE FUNCTION benchmark_task_live_counts()"""
        )


def downgrade() -> None:
    for table in ("benchmark_events", "benchmark_ready_labels"):
        op.execute(f"DROP TRIGGER {table}_count_insert ON {table}")
        op.execute(f"DROP TRIGGER {table}_count_delete ON {table}")
    op.execute("DROP FUNCTION benchmark_task_live_counts()")
    op.drop_column("benchmark_tasks", "live_labels")
    op.drop_column("benchmark_tasks", "live_events")
