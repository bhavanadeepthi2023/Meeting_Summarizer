"""
Centralized application configuration.

All values are overridable via environment variables (or a .env file in the
project root). No secrets are hardcoded — the app will fail fast at startup
if required API keys are missing.
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API credentials ---
    openai_api_key: str = Field(..., description="OpenAI API key, used for Whisper transcription")
    anthropic_api_key: str = Field(..., description="Anthropic API key, used for insight extraction")

    # --- ASR configuration ---
    whisper_model: str = Field(default="whisper-1", description="OpenAI Whisper model name")
    whisper_max_file_size_mb: int = Field(default=24, description="Whisper API hard limit is 25MB; kept at 24 for safety margin")
    whisper_chunk_seconds: int = Field(default=600, description="Duration of each audio chunk sent to Whisper (10 min default)")

    # --- LLM configuration ---
    claude_model: str = Field(default="claude-sonnet-4-6", description="Claude model used for insight extraction")
    claude_max_tokens: int = Field(default=4096)
    llm_extraction_max_retries: int = Field(default=2, description="Retries on schema-validation failure")
    transcript_chars_per_llm_call: int = Field(
        default=60000,
        description="Approx. char budget per Claude call before map-reduce chunking kicks in (~15k tokens, safe margin under context limit)",
    )

    # --- File handling ---
    upload_dir: Path = Field(default=Path("uploads"))
    max_upload_size_mb: int = Field(default=500, description="Hard cap on raw uploaded file size")
    allowed_audio_extensions: tuple[str, ...] = (
        ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac", ".aac",
    )

    # --- Job store ---
    job_retention_hours: int = Field(default=24, description="How long completed job records are kept in memory")

    # --- Server ---
    cors_allow_origins: tuple[str, ...] = ("*",)

    @field_validator("upload_dir")
    @classmethod
    def _ensure_upload_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
