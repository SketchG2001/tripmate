from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )

    app_name: str = Field(default="TripMate", min_length=1)
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    groq_api_key: SecretStr | None = None
    groq_model: str = Field(default="qwen/qwen3.8-27b", min_length=1)
    groq_timeout_seconds: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)
    groq_max_retries: int = Field(default=2, ge=0, le=5)
    checkpoint_db_path: Path = Path("data/runtime/tripmate-checkpoints.sqlite3")
    destination_data_dir: Path = Path("data/destinations")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", min_length=1)
    rag_top_k: int = Field(default=3, gt=0)
    agent_max_tool_calls: int = Field(default=6, gt=0, le=20)

    @field_validator("destination_data_dir", "checkpoint_db_path", mode="after")
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def normalize_optional_key(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
