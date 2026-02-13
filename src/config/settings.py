from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database Configuration
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "intraday_options"
    POSTGRES_HOST: str = "localhost" # Default to localhost, docker-compose will override
    POSTGRES_PORT: int = 5432
    
    # Groww API Configuration
    GROWW_JWT_TOKEN: str
    GROWW_SECRET: str
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    
    # App Settings
    LOG_LEVEL: str = "INFO"
    SYMBOL: str = "NIFTY"
    INTERVAL_SECONDS: int = 60
    SIGNAL_THRESHOLD: float = 0.6
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    class Config:
        env_file = ".env"
        extra = "ignore" 

settings = Settings()
