import os
import sherpa_onnx
import logging
import numpy as np
from typing import Optional
from underthesea import sent_tokenize

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
        decoding_method: str = "greedy_search",
        desired_sr: int = 16000,
        feature_dim: int = 80,
        num_threads: int = 2,
        device: str = "cuda",
    ):
        if self._initialized:
            return

        logger.info(
            "Initializing Zipformer recognizer: repo_id=%s revision=%s provider=%s threads=%s decoding=%s sample_rate=%s feature_dim=%s",
            repo_id,
            revision,
            device,
            num_threads,
            decoding_method,
            desired_sr,
            feature_dim,
        )

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
            decoding_method=decoding_method,
            sample_rate=desired_sr,
            feature_dim=feature_dim,
            debug=False,
            provider=device,
        )

        self.desired_sr = desired_sr
        self._initialized = True
        logger.info("Zipformer recognizer initialized successfully")

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self.desired_sr:
            samples = resample_audio(samples, sample_rate, self.desired_sr)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.desired_sr, samples)
        self.recognizer.decode_stream(stream)
        result_text = stream.result.text

        return result_text
    
    def normalize(self, text: str) -> str:
        sentences = sent_tokenize(text)
        normalized = []
        for s in sentences:
            s = s.strip().capitalize()
            normalized.append(s)
        return " ".join(normalized)

    def warmup(self, duration_sec: float = 0.5) -> None:
        warmup_samples = np.zeros(
            int(self.desired_sr * duration_sec),
            dtype=np.float32,
        )

        logger.info(
            "Running Zipformer warm-up inference with %.3f seconds of silence",
            duration_sec,
        )
        self.transcribe(warmup_samples, self.desired_sr)
        self.normalize("")
        logger.info("Zipformer warm-up completed")

zipformer = Zipformer(
    repo_id=settings.ZIPFORMER_REPO_ID,
    revision=settings.ZIPFORMER_REVISION,
    encoder=settings.ZIPFORMER_ENCODER,
    decoder=settings.ZIPFORMER_DECODER,
    joiner=settings.ZIPFORMER_JOINER,
    tokens=settings.ZIPFORMER_TOKENS,
    decoding_method=settings.ZIPFORMER_DECODING_METHOD,
    desired_sr=settings.ZIPFORMER_SAMPLE_RATE,
    feature_dim=settings.ZIPFORMER_FEATURE_DIM,
    num_threads=settings.ZIPFORMER_NUM_THREADS,
    device="cuda" if settings.USE_CUDA else "cpu",
)
