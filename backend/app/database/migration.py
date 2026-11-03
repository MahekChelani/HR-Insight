"""One-time, idempotent import of local SQLite records into PostgreSQL."""
import logging
from pathlib import Path

from sqlalchemy import MetaData, create_engine, select, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.engine import URL

from app.database.session import Base, DB_PATH, engine
from app.models.db_models import ModelMetadata, PredictionHistory

logger = logging.getLogger(__name__)


def migrate_legacy_sqlite_data(destination_engine=engine, source_path=DB_PATH) -> dict[str, int]:
    """Copy existing prediction and model metadata rows without replacing cloud records."""
    if not Path(source_path).is_file():
        return {}

    source_engine = create_engine(URL.create("sqlite", database=source_path))
    source_metadata = MetaData()
    source_metadata.reflect(bind=source_engine)
    migrated: dict[str, int] = {}
    Base.metadata.create_all(bind=destination_engine)

    try:
        with source_engine.connect() as source, destination_engine.begin() as destination:
            for model in (ModelMetadata, PredictionHistory):
                table_name = model.__tablename__
                source_table = source_metadata.tables.get(table_name)
                if source_table is None:
                    continue

                destination_table = Base.metadata.tables[table_name]
                rows = [
                    dict(row)
                    for row in source.execute(select(source_table)).mappings()
                ]
                source_ids = [row["id"] for row in rows]
                existing_ids = set()
                if source_ids:
                    existing_ids = set(destination.execute(
                        select(destination_table.c.id).where(
                            destination_table.c.id.in_(source_ids)
                        )
                    ).scalars())
                inserted = 0

                for values in rows:
                    if values["id"] in existing_ids:
                        continue
                    if destination_engine.dialect.name == "postgresql":
                        statement = postgresql_insert(destination_table).values(**values)
                        result = destination.execute(
                            statement.on_conflict_do_nothing(
                                index_elements=["id"]
                            ).returning(destination_table.c.id)
                        )
                        if result.scalar_one_or_none() is not None:
                            inserted += 1
                    else:
                        destination.execute(destination_table.insert().values(**values))
                        inserted += 1

                migrated[table_name] = inserted

            if destination_engine.dialect.name == "postgresql":
                for table_name in ("model_metadata", "prediction_history"):
                    destination.execute(text(
                        f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), "
                        f"COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM {table_name}"
                    ))
    finally:
        source_engine.dispose()

    if any(migrated.values()):
        logger.info("Imported legacy SQLite records into PostgreSQL: %s", migrated)
    return migrated
