from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


MigrationFn = Callable[[Connection], None]


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    upgrade: MigrationFn


def _create_schema_with_models(connection: Connection) -> None:
    from backend.core.database import Base

    Base.metadata.create_all(bind=connection)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version="001_initial_schema",
        description="Create initial application schema",
        upgrade=_create_schema_with_models,
    ),
)


def run_migrations(engine: Engine) -> None:
    """
    Uygulama semasini versiyonlu migration kaydi ile gunceller.

    Ilk migration, mevcut Base metadata'sini kullanarak eksik tablolari olusturur
    ve semayi artik yonetilen hale getirir.
    """
    with engine.begin() as connection:
        _ensure_version_table(connection)
        applied_versions = _load_applied_versions(connection)

        for migration in MIGRATIONS:
            if migration.version in applied_versions:
                continue

            migration.upgrade(connection)
            connection.execute(
                text(
                    """
                    INSERT INTO schema_migrations (version, description)
                    VALUES (:version, :description)
                    """
                ),
                {
                    "version": migration.version,
                    "description": migration.description,
                },
            )


def _ensure_version_table(connection: Connection) -> None:
    connection.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
            """
        )
    )


def _load_applied_versions(connection: Connection) -> set[str]:
    rows = connection.execute(
        text("SELECT version FROM schema_migrations ORDER BY version ASC")
    ).fetchall()
    return {row[0] for row in rows}
