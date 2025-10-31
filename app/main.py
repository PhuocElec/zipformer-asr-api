import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.logging import setup_logging
from app.core.settings import settings
from app.api import transcriptions

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    try:
        yield
    finally:
        logger.info("Application shutdown — cleaning up resources")

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
)

app.include_router(transcriptions.router)

logger.info(f"FastAPI app '{settings.APP_NAME}' initialized and routes registered")

@app.get("/health")
def health_check():
    status = {"status": "ok"}
    logger.debug(f"Health check details: {status}")
    return status
