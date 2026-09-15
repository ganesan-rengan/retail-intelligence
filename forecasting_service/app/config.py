"""Service configuration, read from environment variables."""

from functools import lru_cache
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


@lru_cache
def get_settings() -> Settings:
    """Built on first use, not at import time -- so importing this module
    never requires DATABASE_URL unless something actually reads a setting.
    Cached so every caller shares the same instance after that."""
    return Settings()