
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AgriSmart API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"

    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "agrismart_db"
    db_user: str = "agrismart_app"
    db_password: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
