"""Service configuration, read from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Values come from the environment, falling back to .env for local dev."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    models_dir: Path = REPO_ROOT / "models"
    log_level: str = "INFO"


settings = Settings()