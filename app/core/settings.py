from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    # ===== App Config =====
    APP_NAME: str = "zipformer-asr-api"
    LOG_LEVEL: str = "INFO"

    # ===== Auth Config =====
    API_KEYS: Optional[List[str]] = None

    # ===== Model Config =====
    USE_CUDA: bool = True
    HF_TOKEN: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.API_KEYS, str):
            self.API_KEYS = [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

settings = Settings()
