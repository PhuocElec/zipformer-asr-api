import io
import time
import logging
import subprocess
import asyncio
import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, UploadFile, HTTPException, Header, Depends

from app.core.settings import settings
from app.models.zipformer import zipformer

logger = logging.getLogger(__name__)
router = APIRouter()

# --------- Dependencies ---------

async def validate_api_key(api_key: str = Header(None, alias="API-Key")):
    if not settings.API_KEYS:
        return

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")

    if api_key not in settings.API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --------- Router ---------

@router.post("/transcriptions", dependencies=[Depends(validate_api_key)])
async def post_transcription(file: UploadFile = File(...)):
    try:
        data = await file.read()

        samples, sample_rate = await asyncio.to_thread(
            _decode_audio_in_memory,
            data,
            (file.filename or "").lower(),
            (file.content_type or "").lower(),
        )

        start_time = time.monotonic()

        text = await asyncio.to_thread(zipformer.transcribe, samples, sample_rate)
        text = await asyncio.to_thread(zipformer.normalize, text)

        logger.info(
            "Transcription completed in %.3f seconds",
            time.monotonic() - start_time,
        )

        return {"text": text}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during transcription")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


# --------- Helpers ---------

def _decode_audio_in_memory(
    data: bytes,
    filename: str,
    content_type: str,
    max_bytes: int = 1 * 1024 * 1024,  # 1 MB
) -> tuple[np.ndarray, int]:
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    filename = filename.lower()
    content_type = content_type.lower()

    is_wav = (
        "audio/wav" in content_type
        or "audio/x-wav" in content_type
        or filename.endswith(".wav")
    )

    is_compressed = (
        # mp3
        "audio/mpeg" in content_type
        or "audio/mp3" in content_type
        or filename.endswith(".mp3")
        # opus / ogg / webm
        or "audio/ogg" in content_type
        or "audio/opus" in content_type
        or "audio/webm" in content_type
        or filename.endswith((".opus", ".ogg", ".webm"))
    )

    if is_wav:
        return _decode_wav_from_bytes(data)

    if is_compressed:
        return _decode_with_ffmpeg(data)

    raise HTTPException(
        status_code=400,
        detail="Only wav, mp3, opus, ogg, webm audio files are supported",
    )


def _decode_wav_from_bytes(buf: bytes) -> tuple[np.ndarray, int]:
    with sf.SoundFile(io.BytesIO(buf)) as f:
        samples = f.read(dtype="float32")
        sr = f.samplerate
    
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
        
    return samples, sr


def _decode_with_ffmpeg(buf: bytes) -> tuple[np.ndarray, int]:
    """
    Decode any compressed audio (mp3, opus, ogg, webm)
    → mono, 16kHz, float32
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v", "error",
                "-i", "pipe:0",
                "-ac", "1",          # mono
                "-ar", "16000",      # resample
                "-f", "f32le",       # float32 raw
                "pipe:1",
            ],
            input=buf,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found on server")
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=400,
            detail=f"ffmpeg decode failed: {e.stderr.decode(errors='ignore')}",
        )

    samples = np.frombuffer(result.stdout, dtype=np.float32)

    if samples.size == 0:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")

    return samples, 16000
