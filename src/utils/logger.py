"""
src/utils/logger.py
────────────────────
Centralized logging setup for the project.
Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


_LOG_DIR = Path(__file__).resolve().parents[2] / "results" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger that writes to both console and a daily log file.

    Args:
        name:  Module name — use __name__ when calling.
        level: Logging level (default INFO).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Formatter ──────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File handler (daily rotating file) ────────────────────
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = _LOG_DIR / f"vf_{today}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logger initialized successfully")
    log.warning("This is a warning example")
    log.error("This is an error example")
