from sqlalchemy import create_engine, inspect

from app.db import init_db
from app.models import CropRecord  # noqa: F401


def test_initialize_database_creates_model_tables(monkeypatch):
    engine = create_engine("sqlite://")
    monkeypatch.setattr(init_db, "engine", engine)

    init_db.initialize_database()

    assert "crop_records" in inspect(engine).get_table_names()
    engine.dispose()
