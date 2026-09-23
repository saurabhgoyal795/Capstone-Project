"""Logging configuration: console + rotating file handler."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(threadName)-18s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are noisy at INFO level.
_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "google", "pypdf")


def setup_logging(level: str = "INFO", log_dir: Path = Path("logs")) -> Path:
    """Configure the root logger with a console handler and a rotating file handler.

    Args:
        level: Log level name (e.g. "INFO", "DEBUG").
        log_dir: Directory where ``app.log`` is written (created if missing).

    Returns:
        Path to the log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Idempotent: remove handlers from previous calls (e.g. repeated CLI runs in tests).
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(numeric_level)
    root.addHandler(console)

    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file
