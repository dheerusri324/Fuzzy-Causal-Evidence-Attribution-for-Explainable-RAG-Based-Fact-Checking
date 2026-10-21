"""
Logging setup for the project.

Provides a configured logger with console output.
Usage:
    from utils.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import sys


_CONFIGURED = False


def setup_logging(level: str = "INFO", debug: bool = False) -> None:
    """Configure root logging. Call once at application startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Configures root logging if not yet done."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
