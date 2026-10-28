
from sqlalchemy import text

from app.db.database import Base, engine
from app.models import CropRecord  # noqa: F401


def initialize_database() -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    Base.metadata.create_all(bind=engine)
    print("Database connection successful.")
    print("AgriSmart database tables initialized.")


if __name__ == "__main__":
    initialize_database()
