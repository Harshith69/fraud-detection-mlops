"""Thin MongoDB helper for ingest and read-back.

The pymongo import is deferred so projects that don't use Mongo (and tests
that mock it) can still import this module without the dependency present.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, TYPE_CHECKING

from ..config import MongoConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pymongo.collection import Collection


@contextmanager
def mongo_collection(cfg: MongoConfig) -> Iterator["Collection"]:
    """Yield a pymongo ``Collection`` for ``cfg`` and ensure the client closes.

    Raises:
        RuntimeError: when ``cfg`` is missing required values.
        ImportError: when ``pymongo`` is not installed.
    """

    if not cfg.is_configured:
        raise RuntimeError(
            "MongoDB is not configured. Set MONGODB_URI, MONGODB_DB and "
            "MONGODB_COLLECTION in .env."
        )

    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - import error path
        raise ImportError("pymongo is required for MongoDB operations") from exc

    client = MongoClient(cfg.uri, serverSelectionTimeoutMS=10_000)
    try:
        # Force a server round-trip so connection problems surface here, not
        # later inside an unrelated stage.
        client.admin.command("ping")
        collection = client[cfg.database][cfg.collection]
        yield collection
    finally:
        client.close()
