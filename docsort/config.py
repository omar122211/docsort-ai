"""Settings, read from environment variables or a local .env file (never committed)."""
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini API (free tier) — used for field extraction and the zero-shot comparison
    gemini_api_key: SecretStr | None = None
    vision_model: str = "gemini-3.5-flash-lite"
    fallback_vision_model: str | None = "gemini-3.8-flash"

    # Classifier
    data_dir: Path = ROOT / "data"
    model_path: Path = ROOT / "models" / "docsort-classifier.pt"
    image_size: int = 320
    min_confidence: float = 0.55  # below this the document is reported as "unsure"

    # Upload limits
    max_file_mb: int = 8
    max_files_per_batch: int = 20
    rate_limit_per_minute: int = 30

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
