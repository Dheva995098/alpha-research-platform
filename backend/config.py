"""
Configuration module for Alpha Research Platform.
Handles environment variables, database setup, and app settings.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings loaded from .env or environment variables."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    # FastAPI
    environment: str = "development"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Database
    database_url: str = "sqlite:///./alpha_research.db"
    
    # Security
    secret_key: str = "dev-secret-key-change-in-production"
    aes_key: Optional[str] = None
    
    # API Keys
    openai_api_key: Optional[str] = None
    claude_api_key: Optional[str] = None
    
    # Slack
    slack_webhook_url: Optional[str] = None
    
    # SendGrid
    sendgrid_api_key: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"

    # BRAIN API pacing
    brain_request_interval_seconds: float = 5.0
    brain_submit_interval_seconds: float = 55.0
    brain_submit_global_interval_seconds: float = 25.0
    brain_rate_limit_cooldown_seconds: int = 120
    brain_poll_account_gap_seconds: int = 10
    single_account_mode: bool = True
    primary_account_id: Optional[int] = 1

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        """Accept booleans plus common environment mode strings."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "production", "prod"}:
                return False
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value):
        """Accept hosted Postgres URLs from common providers."""
        if isinstance(value, str) and value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        if isinstance(value, str) and value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value
    

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
