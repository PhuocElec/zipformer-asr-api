from pydantic import Json
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    # ===== App Config =====
    APP_NAME: str = "zipformer-asr-api"
    LOG_LEVEL: str = "INFO"

    # ===== Auth Config =====
    API_KEYS: Optional[Json[List[str]]] = []

    # ===== Model Config =====
    USE_CUDA: bool = True
    HF_TOKEN: Optional[str] = None

    ZIPFORMER_REPO_ID: str = "zipformer/zipformer-asr-base"
    ZIPFORMER_REVISION: Optional[str] = None
    ZIPFORMER_ENCODER: str = "encoder.onnx"
    ZIPFORMER_DECODER: str = "decoder.onnx"
    ZIPFORMER_JOINER: str = "joiner.onnx"
    ZIPFORMER_TOKENS: str = "tokens.txt"
    ZIPFORMER_NUM_THREADS: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
