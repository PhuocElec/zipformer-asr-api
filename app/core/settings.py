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

    ZIPFORMER_REPO_ID: str = "zzasdf/viet_iter3_pseudo_label"
    ZIPFORMER_REVISION: Optional[str] = "main"
    ZIPFORMER_ENCODER: str = "exp/encoder-epoch-12-avg-8.onnx"
    ZIPFORMER_DECODER: str = "exp/decoder-epoch-12-avg-8.onnx"
    ZIPFORMER_JOINER: str = "exp/joiner-epoch-12-avg-8.onnx"
    ZIPFORMER_TOKENS: str = "data/Vietnam_bpe_2000_new/tokens.txt"
    ZIPFORMER_NUM_THREADS: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
