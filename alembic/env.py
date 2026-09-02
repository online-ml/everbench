from __future__ import annotations

import os

from sqlalchemy import engine_from_config, pool

from alembic import context
from everbench.db import sqlalchemy_url
from everbench.schema import Base

config = context.config
if url := os.getenv("DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", sqlalchemy_url(url))
else:
    raise RuntimeError(
        "DATABASE_URL must be set before running Alembic, for example: postgresql://USER:PASSWORD@HOST:5432/everbench"
    )
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
