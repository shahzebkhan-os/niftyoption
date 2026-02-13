import logging
from typing import Optional, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure structured logging for config phase
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Production-grade configuration management using Pydantic.
    Enforces type safety, presence of required fields, and environment-specific overrides.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- Core Environment ---
    ENV: Literal["dev", "prod", "test"] = "dev"
    DRY_RUN: bool = True
    LOG_LEVEL: str = "INFO"
    
    # --- Database Configuration ---
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "intraday_options"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    # Groww API Configuration
    GROWW_JWT_TOKEN: str
    GROWW_SECRET: str
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    
    # App Settings
    SYMBOLS: list[str] = ["NIFTY", "BANKNIFTY"]
    INTERVAL_SECONDS: int = 60
    SIGNAL_THRESHOLD: float = 0.6
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()
