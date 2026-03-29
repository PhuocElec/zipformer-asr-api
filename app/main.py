import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logging import setup_logging
from app.core.settings import settings

setup_logging()
logger = logging.getLogger(__name__)

from app.api import transcriptions
from app.models.zipformer import zipformer


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    try:
        warmup_started_at = time.monotonic()
        await asyncio.to_thread(zipformer.warmup)
        logger.info(
            "Application warm-up completed in %.3f seconds",
            time.monotonic() - warmup_started_at,
        )
        yield
    finally:
        logger.info("Application shutdown - cleaning up resources")


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
)

app.include_router(transcriptions.router)

logger.info("FastAPI app '%s' initialized and routes registered", settings.APP_NAME)


@app.get("/health")
def health_check():
    status = {"status": "ok"}
    logger.debug("Health check details: %s", status)
    return status
