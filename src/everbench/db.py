"""Shared database configuration for workers and the API."""

from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def database_url() -> str:
    try:
        return os.environ["DATABASE_URL"]
    except KeyError as error:
        raise RuntimeError("DATABASE_URL must be set") from error


def make_engine(*, url: str | None = None) -> Engine:
    value = url or database_url()
    if not value.startswith("sqlite:"):
        raise ValueError("DATABASE_URL must point to a SQLite database")
    engine = create_engine(
        value,
        pool_size=int(os.getenv("EVERBENCH_DB_POOL_SIZE", "10")),
        max_overflow=0,
        connect_args={"timeout": 120, "check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(connection, _record) -> None:  # noqa: PLR0917 -- SQLAlchemy callback
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=120000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def make_session_factory(*, url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(make_engine(url=url), expire_on_commit=False)


def lock_transaction(*, session: Session, name: str) -> None:
    """Acquire SQLite's writer lock before reading state that will be changed."""
    from everbench.schema import DatabaseLock

    session.execute(
        insert(DatabaseLock).values(name=name).on_conflict_do_update(index_elements=["name"], set_={"name": name})
    )


def allocate_sequence(*, session: Session, name: str, count: int) -> range:
    """Reserve a durable global sequence range in one SQLite write."""
    from everbench.schema import DatabaseSequence

    if count < 1:
        return range(0)
    end = session.scalar(
        insert(DatabaseSequence)
        .values(name=name, value=count)
        .on_conflict_do_update(
            index_elements=["name"],
            set_={"value": DatabaseSequence.value + count},
        )
        .returning(DatabaseSequence.value)
    )
    assert end is not None
    return range(end - count + 1, end + 1)
