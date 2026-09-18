"""Structured JSON logging (plan §10): one JSON object per line to stdout,
which is what a container platform (Render) captures as-is -- no log
aggregation service dependency needed.
"""
from __future__ import annotations

import json
import logging
import sys

_LOGGER_NAME = "polyo"


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # The message *is* the JSON payload (built by `log_event` below) --
        # the formatter only needs to emit it as-is, not wrap it again.
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))
