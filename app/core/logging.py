import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from app.core.settings import settings


class TZFormatter(logging.Formatter):
    """Custom formatter to include timezone-aware timestamps with milliseconds."""
    def __init__(self, fmt=None, datefmt=None, tz_name="Asia/Ho_Chi_Minh"):
        super().__init__(fmt, datefmt)
        try:
            self.tz = ZoneInfo(tz_name)
        except Exception:
            self.tz = ZoneInfo("UTC")

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=self.tz)
        if datefmt:
            s = dt.strftime(datefmt)
            return f"{s}.{int(record.msecs):03d}"
        return dt.isoformat(timespec="milliseconds")


def setup_logging():
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = TZFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz_name=getattr(settings, "TIMEZONE", "Asia/Ho_Chi_Minh"),
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    root_logger.info("Logging settingsured successfully.")
