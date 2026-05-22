"""Project-wide logger configuration.

A single helper so every CLI stage has the same output format and level can
be controlled with the ``LOG_LEVEL`` environment variable.
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Configuration is applied once on the root logger; child loggers inherit it.
    """

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return logging.getLogger(name)
