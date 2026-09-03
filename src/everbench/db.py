"""Shared database configuration for workers and the API."""

from __future__ import annotations

import os
from hashlib import blake2b

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def database_url() -> str:
    try:
        return os.environ["DATABASE_URL"]
    except KeyError as error:
        raise RuntimeError("DATABASE_URL must be set") from error


def sqlalchemy_url(url: str | None = None) -> str:
    """Use psycopg 3 for ordinary Postgres and Railway connection URLs."""
    value = url or database_url()
    if value.startswith("postgres://"):
        value = "postgresql://" + value.removeprefix("postgres://")
    if value.startswith("postgresql://"):
        value = "postgresql+psycopg://" + value.removeprefix("postgresql://")
    return value


def make_engine(url: str | None = None) -> Engine:
    # Each Railway worker is single-process. Keep enough connections for its
    # collectors, compactor, heartbeat, and connection-pinned learners.
    return create_engine(
        sqlalchemy_url(url),
        pool_pre_ping=True,
        # Each learner pins one connection while holding its advisory lock.
        # Leave capacity for collectors, heartbeats, and archive work.
        pool_size=int(os.getenv("EVERBENCH_DB_POOL_SIZE", "10")),
        max_overflow=0,
    )


def make_session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(make_engine(url), expire_on_commit=False)


def advisory_key(*parts: str) -> int:
    """Map a namespaced application identity onto PostgreSQL's signed bigint keyspace."""
    digest = blake2b("\0".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)
