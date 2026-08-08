"""Alembic environment configuration.

Reads the database URL from the Indus Kernel settings (env vars).
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import the Base from ik_kernel's models (will be created in M1)
# from ik_kernel.models import Base
# target_metadata = Base.metadata

config = context.config

# Override the URL from env if present
if "INDUS_DATABASE_URL" in os.environ:
    db_url = os.environ["INDUS_DATABASE_URL"]
    # Alembic wants sync URL (no +asyncpg)
    db_url = db_url.replace("+asyncpg", "")
    config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None  # Will be set in M1 when Base is defined


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without connecting)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to DB and apply)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
