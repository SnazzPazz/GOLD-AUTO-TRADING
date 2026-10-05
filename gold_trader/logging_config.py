"""Audit-trail logging: every signal and every simulated/demo trade gets
written to logs/gold_trader.log as well as the console, so the demo run
can be reviewed and verified after the fact rather than trusted blindly.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("gold_trader")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(LOG_DIR / "gold_trader.log", maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
