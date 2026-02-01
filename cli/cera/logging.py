"""Structured logging configuration for CERA pipeline."""

import sys
import logging

import structlog


def configure_logging(debug: bool = False):
    """
    Configure structlog for the CERA pipeline.

    Uses pretty-printed output for TTY (development) and
    JSON output for non-TTY (production/Docker).

    Args:
        debug: Enable debug-level logging
    """
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure standard library logging (for structlog integration)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
    ]

    if sys.stderr.isatty():
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with the given name.

    Args:
        name: Logger name (e.g., "cera.llm.openrouter")

    Returns:
        Configured structlog BoundLogger
    """
    return structlog.get_logger(name)
