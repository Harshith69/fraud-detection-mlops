"""Stage 1: ingest the raw CSV into MongoDB and snapshot it for downstream stages.

The Kaggle CSV is treated as the source of truth. Each pipeline run:

1. Reads the CSV from ``params.paths.raw_csv`` and validates its schema.
2. (Optional) Pushes the rows into MongoDB so they are durable and queryable.
3. Writes a snapshot CSV (``params.paths.raw_snapshot``) which is the file
   downstream stages depend on. DVC tracks this snapshot so any change to the
   source data invalidates the cached pipeline outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ..config import Config, get_config
from ..logging_utils import get_logger
from .mongo_client import mongo_collection
from .schema import EXPECTED_COLUMNS, validate_dataframe

_LOG = get_logger(__name__)


@dataclass
class IngestResult:
    """Summary of what an ingest run produced."""

    rows: int
    snapshot_path: Path
    pushed_to_mongo: bool
    mongo_inserted: int


def _chunked(records: List[dict], size: int) -> Iterable[List[dict]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def push_to_mongodb(df: pd.DataFrame, cfg: Config) -> int:
    """Insert ``df`` rows into MongoDB. Returns number of inserted documents."""

    ingest_cfg = cfg.section("ingest")
    batch_size = int(ingest_cfg.get("mongo_batch_size", 1000))
    drop_existing = bool(ingest_cfg.get("mongo_drop_existing", True))

    records = df.to_dict(orient="records")
    inserted = 0
    with mongo_collection(cfg.mongo) as collection:
        if drop_existing:
            _LOG.info(
                "Dropping existing collection %s.%s", cfg.mongo.database, cfg.mongo.collection
            )
            collection.drop()

        for batch in _chunked(records, batch_size):
            if not batch:
                continue
            result = collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)

    _LOG.info("Inserted %d documents into MongoDB", inserted)
    return inserted


def run_ingest(cfg: Config | None = None) -> IngestResult:
    """Ingest the raw CSV, optionally push to Mongo, and write the snapshot."""

    cfg = cfg or get_config()
    raw_csv = cfg.path("raw_csv")
    snapshot = cfg.path("raw_snapshot")
    ingest_cfg = cfg.section("ingest")

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {raw_csv}. Place the Kaggle dataset there before running ingest."
        )

    _LOG.info("Reading raw CSV from %s", raw_csv)
    df = pd.read_csv(raw_csv)

    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise ValueError("Raw dataset failed schema validation: " + "; ".join(validation.errors))

    pushed = False
    inserted = 0
    if ingest_cfg.get("push_to_mongo", True):
        if cfg.mongo.is_configured:
            inserted = push_to_mongodb(df, cfg)
            pushed = True
        else:
            _LOG.warning("push_to_mongo is True but MongoDB is not configured; skipping push.")

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot, index=False, columns=[c for c in EXPECTED_COLUMNS if c in df.columns])
    _LOG.info("Wrote snapshot CSV to %s (%d rows)", snapshot, len(df))

    return IngestResult(
        rows=len(df),
        snapshot_path=snapshot,
        pushed_to_mongo=pushed,
        mongo_inserted=inserted,
    )


def main() -> None:
    """CLI entry point used by the DVC stage."""
    result = run_ingest()
    _LOG.info(
        "Ingest complete: rows=%d snapshot=%s pushed=%s inserted=%d",
        result.rows,
        result.snapshot_path,
        result.pushed_to_mongo,
        result.mongo_inserted,
    )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
