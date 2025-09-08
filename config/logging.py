import os
import logging
import logging.config
from datetime import datetime
from pathlib import Path

# Determine log directory (./logs/<timestamp>/)
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_logs_root = Path(__file__).resolve().parent.parent / "logs" / _timestamp
_logs_root.mkdir(parents=True, exist_ok=True)
_info_log_path = _logs_root / "info.json"
_error_log_path = _logs_root / "error.json"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {  # New formatter for detailed logs
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s",
        },
        "json": {
            "format": "%(asctime)s %(process)d %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.json.JsonFormatter",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json",
        },
        "info_file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": str(_info_log_path),
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "json",
            "filename": str(_error_log_path),
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "info_file", "error_file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)