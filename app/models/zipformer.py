import os
import sherpa_onnx
import logging
import numpy as np
from typing import Optional

from app.core.settings import settings
from app.models.downloader import download_hf_model
from app.utils.audio import resample_audio

logger = logging.getLogger(__name__)

class Zipformer:
    _instance: Optional["Zipformer"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        repo_id: str,
        revision: str,
        encoder: str,
        decoder: str,
        joiner: str,
        tokens: str,
        desired_sr: int = 16000,
        num_threads: int = 2,
        device: str = "cuda",
    ):
        if self._initialized:
            return
        
        model_dir = download_hf_model(
            repo_id,
            revision=revision,
            token=settings.HF_TOKEN,
        )

        encoder_path = os.path.join(model_dir, encoder)
        decoder_path = os.path.join(model_dir, decoder)
        joiner_path  = os.path.join(model_dir, joiner)
        tokens_path  = os.path.join(model_dir, tokens)

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=num_threads,
            decoding_method="greedy_search",
            sample_rate=desired_sr,
            feature_dim=80,
            debug=False,
            provider=device,
        )

        self.desired_sr = desired_sr
        self._initialized = True

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.desired_sr:
            samples = resample_audio(samples, sample_rate, self.desired_sr)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.desired_sr, samples)
        self.recognizer.decode_stream(stream)
        result_text = stream.result.text

        return result_text

zipformer = Zipformer(
    repo_id=settings.ZIPFORMER_REPO_ID,
    revision=settings.ZIPFORMER_REVISION,
    encoder=settings.ZIPFORMER_ENCODER,
    decoder=settings.ZIPFORMER_DECODER,
    joiner=settings.ZIPFORMER_JOINER,
    tokens=settings.ZIPFORMER_TOKENS,
    num_threads=settings.ZIPFORMER_NUM_THREADS,
    device="cuda" if settings.USE_CUDA else "cpu",
)
