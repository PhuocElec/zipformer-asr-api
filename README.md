# zipformer-asr-api

A lightweight HTTP speech-to-text (ASR) server built with FastAPI that serves a Zipformer-based model via `sherpa-onnx` and ONNX Runtime.

## Prerequisites

- Python 3.10 or newer installed and on your PATH
- FFmpeg installed (required for decoding MP3 uploads)
- For GPU acceleration (default requirements):
  - NVIDIA GPU + recent driver
  - CUDA 12.x compatible stack (the pinned wheels target CUDA 12 + cuDNN 9)

Note for CPU-only systems: the provided `requirements.txt` pins GPU builds. If you do not have a CUDA-capable GPU, consider replacing `onnxruntime-gpu` with `onnxruntime` and using a CPU build of `sherpa-onnx`, or install from a custom constraints file that targets CPU.

## Installation

1) Create and activate a virtual environment

Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash):

```
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

## Configuration

1) Create your environment file from the example and adjust values as needed:

Windows (PowerShell):

```
Copy-Item .env.example .env
```

macOS/Linux (bash):

```
cp .env.example .env
```

2) Open `.env` and set the variables that matter for your setup. Common options:

- `APP_NAME`: Application name shown in logs/docs.
- `LOG_LEVEL`: Logging level (e.g., `INFO`, `DEBUG`).
- `USE_CUDA`: `true` to use GPU, `false` for CPU.
- `HF_TOKEN`: Hugging Face token if the model requires authentication.
- `ZIPFORMER_REPO_ID`: HF repo id to download model weights.
- `ZIPFORMER_*`: Filenames and threads for the ONNX model components.

The app loads configuration from `.env` automatically.

## Run the HTTP server

Basic run:

```
uvicorn app.main:app --port 8000
```

Recommended for development (auto-reload):

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:

- Health check: `GET http://127.0.0.1:8000/health`
- OpenAPI/Swagger UI: `http://127.0.0.1:8000/docs`

## Run with Docker

Run the API using Docker Compose:

```
cd zipformer-asr-api
docker compose -f docker/docker-compose.yml up -d
```

The service exposes port `8000` by default. Visit `http://127.0.0.1:8000/docs`.

## Usage example

Transcribe a short WAV/MP3 file via `curl`:

```
curl -X POST \
  -F "file=@sample.wav" \
  http://127.0.0.1:8000/transcriptions
```

Response:

```
{"text": "recognized text here"}
```

Notes:

- MP3 uploads require FFmpeg available on the server.
- Very large files are rejected; keep uploads reasonably small.

## Troubleshooting

- FFmpeg not found: install FFmpeg and ensure it is on your PATH.
- GPU package install errors: ensure a compatible NVIDIA driver and CUDA 12.x toolchain; otherwise switch to CPU packages as noted above.
- HF model download issues: if the model is private, set `HF_TOKEN` in `.env`. For offline use, pre-populate `app/weights` with the model files.

## License

See `LICENSE` for details.
