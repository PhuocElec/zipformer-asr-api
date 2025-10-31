from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    # ===== App Config =====
    APP_NAME: str = "zipformer-asr-api"
    LOG_LEVEL: str = "INFO"

    # ===== Auth Config =====
    API_KEYS: Optional[str] = None

    # ===== Model Config =====
    USE_CUDA: bool = True
    HF_TOKEN: Optional[str] = None

    ZIPFORMER_REPO_ID: str = "zipformer/zipformer-asr-base"
    ZIPFORMER_REVISION: Optional[str] = None
    ZIPFORMER_ENCODER: str = "encoder.onnx"
    ZIPFORMER_DECODER: str = "decoder.onnx"
    ZIPFORMER_JOINER: str = "joiner.onnx"
    ZIPFORMER_TOKENS: str = "tokens.txt"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.API_KEYS, str):
            keys = [k.strip() for k in self.API_KEYS.split(",") if k.strip()]
            self.API_KEYS = keys if keys else None

settings = Settings()
