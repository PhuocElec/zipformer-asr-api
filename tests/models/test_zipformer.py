import soundfile as sf

from app.models.zipformer import zipformer

audio_path = r"tests/models/test.mp3"
samples, sample_rate = sf.read(audio_path, dtype="float32")
text = zipformer.transcribe(samples, sample_rate)

print("Result:", text)