
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_name: str = "AgriSmart API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"

    # MySQL database settings
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "agrismart_db"
    db_user: str = "agrismart_app"
    db_password: str = ""

    # Google Gemini API settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # Environment configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
