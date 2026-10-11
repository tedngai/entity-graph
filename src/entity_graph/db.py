"""PostgreSQL connection helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config, load_config

log = logging.getLogger(__name__)


def connect(config: Config | None = None):
    """Open a psycopg connection. Heavy import is lazy on purpose."""
    import psycopg

    config = config or load_config()
    return psycopg.connect(config.dsn)


def apply_schema(config: Config | None = None) -> None:
    """Apply db/schema.sql (idempotent: CREATE ... IF NOT EXISTS)."""
    config = config or load_config()
    schema_file = Path(config.schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(
            f"Schema not found at {schema_file}. Run CLI commands from the repo root."
        )
    migrations = sorted((schema_file.parent / "migrations").glob("*.sql"))
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(schema_file.read_text(encoding="utf-8"))
        for path in migrations:
            cur.execute(path.read_text(encoding="utf-8"))
            log.info("Applied migration %s", path.name)
    log.info("Schema applied from %s", schema_file)
