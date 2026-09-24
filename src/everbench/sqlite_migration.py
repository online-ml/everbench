"""One-time copy of the retained benchmark from Postgres to SQLite."""

from __future__ import annotations

import logging

from sqlalchemy import Engine, MetaData, func, select, text
from sqlalchemy.dialects.sqlite import insert

from everbench.schema import Base, DatabaseMigration, DatabaseSequence

SKIPPED_TABLES = {"database_locks", "database_migrations", "database_sequences", "worker_heartbeats"}
REMOVED_TASK = "citibike"


def _rows_for_retained_tasks(*, source_table, target_table):
    statement = select(*(source_table.c[column.name] for column in target_table.columns))
    if "task_name" in source_table.c:
        statement = statement.where(source_table.c.task_name != REMOVED_TASK)
    return statement


def _prune_artifacts(*, target: Engine) -> None:
    with target.begin() as connection:
        connection.execute(
            text(
                """DELETE FROM model_artifacts
                    WHERE artifact_id NOT IN (
                        SELECT artifact_id FROM benchmark_models WHERE artifact_id IS NOT NULL
                        UNION SELECT artifact_id FROM model_snapshots
                        UNION SELECT champion_artifact_id FROM auto_experiments
                        UNION SELECT candidate_artifact_id FROM auto_experiments WHERE candidate_artifact_id IS NOT NULL
                    )"""
            )
        )


def sync_sequence_floors(*, source: Engine, target: Engine) -> None:
    """Keep new IDs above every old global ID and retained model checkpoint."""
    metadata = MetaData()
    metadata.reflect(
        bind=source,
        only=["benchmark_events", "benchmark_ready_labels", "benchmark_models", "model_snapshots", "auto_experiments"],
    )
    columns = (
        (
            "events",
            ("benchmark_events", "sequence"),
            ("benchmark_models", "start_sequence"),
            ("benchmark_models", "prediction_cursor_sequence"),
            ("auto_experiments", "comparison_end_sequence"),
        ),
        (
            "ready_labels",
            ("benchmark_ready_labels", "sequence"),
            ("benchmark_models", "label_cursor_sequence"),
            ("model_snapshots", "checkpoint_ready_sequence"),
        ),
    )
    with source.connect() as connection:
        floors = {
            name: max(
                int(connection.scalar(select(func.coalesce(func.max(metadata.tables[table].c[column]), 0))) or 0)
                for table, column in references
            )
            for name, *references in columns
        }
    with target.begin() as connection:
        for name, floor in floors.items():
            if floor:
                connection.execute(
                    insert(DatabaseSequence)
                    .values(name=name, value=floor)
                    .on_conflict_do_update(
                        index_elements=["name"], set_={"value": func.max(DatabaseSequence.value, floor)}
                    )
                )


def copy_postgres_to_sqlite(*, source: Engine, target: Engine) -> dict[str, int]:
    """Copy one stopped Postgres snapshot, excluding the removed task.

    The target must be empty. A failed copy leaves Postgres untouched and must
    be retried with a fresh SQLite volume so partial data cannot masquerade as
    a complete migration.
    """
    if source.dialect.name != "postgresql" or target.dialect.name != "sqlite":
        raise ValueError("expected a Postgres source and SQLite target")
    Base.metadata.create_all(target)
    with target.connect() as connection:
        for table in Base.metadata.sorted_tables:
            if connection.execute(select(table).limit(1)).first() is not None:
                raise ValueError(f"SQLite target already contains data in {table.name}")

    copied: dict[str, int] = {}
    source_metadata = MetaData()
    source_metadata.reflect(
        bind=source, only=[table.name for table in Base.metadata.sorted_tables if table.name not in SKIPPED_TABLES]
    )
    with source.connect().execution_options(stream_results=True) as source_connection:
        for table in Base.metadata.sorted_tables:
            if table.name in SKIPPED_TABLES:
                continue
            source_rows = source_connection.execute(
                _rows_for_retained_tasks(source_table=source_metadata.tables[table.name], target_table=table)
            ).mappings()
            count = 0
            while batch := source_rows.fetchmany(100):
                with target.begin() as target_connection:
                    target_connection.execute(table.insert(), [dict(row) for row in batch])
                count += len(batch)
            copied[table.name] = count
            logging.info("copied %s: %d rows", table.name, count)

    _prune_artifacts(target=target)
    sync_sequence_floors(source=source, target=target)
    with target.begin() as connection:
        foreign_key_errors = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(f"SQLite has {len(foreign_key_errors)} foreign key errors")
        if connection.exec_driver_sql("PRAGMA quick_check").scalar() != "ok":
            raise RuntimeError("SQLite integrity check failed")
        connection.execute(insert(DatabaseMigration).values(name="postgres-import"))
    with target.connect() as connection:
        connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
    return copied
