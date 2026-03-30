import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger that writes to
    both the console and a log file inside logs/

    Args:
        name: usually pass __name__ from the calling file

    Returns:
        configured logger instance
    """

    # ── Create logs/ directory if it doesn't exist ──
    os.makedirs("logs", exist_ok=True)

    # ── Log filename based on today's date ──────────
    # e.g. logs/pranalyzer_2024-01-15.log
    log_filename = f"logs/pranalyzer_{datetime.now().strftime('%Y-%m-%d')}.log"

    # ── Logger instance ─────────────────────────────
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger
    # already exists (happens on re-imports)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Format ──────────────────────────────────────
    formatter = logging.Formatter(
        fmt   = "%(asctime)s | %(levelname)-8s | %(filename)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    # ── Console handler (INFO and above) ────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler (DEBUG and above) ──────────────
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ── Attach both handlers ─────────────────────────
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    print("Check logs/ folder for the log file!")