from pydantic import Json, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    # ===== App Config =====
    APP_NAME: str = "zipformer-asr-api"
    LOG_LEVEL: str = "INFO"
    PORT: int = 8000
    WORKERS: int = 1

    # ===== Auth Config =====
    API_KEYS: Optional[Json[List[str]]] = Field(default="[]")

    # ===== Model Config =====
    USE_CUDA: bool = True
    HF_TOKEN: Optional[str] = None

    ZIPFORMER_REPO_ID: str = "hynt/Zipformer-30M-RNNT-6000h"
    ZIPFORMER_REVISION: Optional[str] = "main"
    ZIPFORMER_ENCODER: str = "encoder-epoch-20-avg-10.onnx"
    ZIPFORMER_DECODER: str = "decoder-epoch-20-avg-10.onnx"
    ZIPFORMER_JOINER: str = "joiner-epoch-20-avg-10.onnx"
    ZIPFORMER_TOKENS: str = "config.json"
    ZIPFORMER_NUM_THREADS: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
