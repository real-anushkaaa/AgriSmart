
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_name: str = "AgriSmart API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"

    # MySQL database settings
    database_url: str | None = None
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "agrismart_db"
    db_user: str = "agrismart_app"
    db_password: str = ""

    # Google Gemini API settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # API safety and browser client settings
    api_access_token: str = ""
    cors_origins: str = ""
    max_image_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    # Environment configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
