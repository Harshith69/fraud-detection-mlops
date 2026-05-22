"""Tests for the ingest stage (Mongo client mocked)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from fraud_detection.data import ingest as ingest_module
from fraud_detection.data.schema import EXPECTED_COLUMNS


def _setup_raw_csv(cfg, df: pd.DataFrame) -> Path:
    raw = cfg.path("raw_csv")
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)
    return raw


def test_run_ingest_writes_snapshot_without_mongo(tmp_path: Path, make_config, synthetic_df):
    cfg = make_config(
        tmp_path,
        params_overrides={
            "ingest": {
                "push_to_mongo": False,
                "mongo_batch_size": 10,
                "mongo_drop_existing": False,
            }
        },
    )
    _setup_raw_csv(cfg, synthetic_df)

    result = ingest_module.run_ingest(cfg)

    assert result.rows == len(synthetic_df)
    assert result.pushed_to_mongo is False
    assert result.snapshot_path.exists()

    snapshot = pd.read_csv(result.snapshot_path)
    # Snapshot should contain only the expected columns and preserve row count.
    assert list(snapshot.columns) == [c for c in EXPECTED_COLUMNS if c in synthetic_df.columns]
    assert len(snapshot) == len(synthetic_df)


def test_run_ingest_skips_mongo_when_unconfigured(
    tmp_path: Path, make_config, synthetic_df, caplog
):
    cfg = make_config(
        tmp_path,
        params_overrides={
            "ingest": {
                "push_to_mongo": True,
                "mongo_batch_size": 10,
                "mongo_drop_existing": False,
            }
        },
    )
    _setup_raw_csv(cfg, synthetic_df)

    with caplog.at_level("WARNING"):
        result = ingest_module.run_ingest(cfg)

    assert result.pushed_to_mongo is False
    assert any("not configured" in record.message for record in caplog.records)


def test_run_ingest_pushes_to_mongo_when_configured(
    tmp_path: Path, make_config, synthetic_df, monkeypatch
):
    cfg = make_config(
        tmp_path,
        params_overrides={
            "ingest": {
                "push_to_mongo": True,
                "mongo_batch_size": 50,
                "mongo_drop_existing": True,
            }
        },
    )
    # Pretend Mongo is configured. The actual client is mocked below.
    object.__setattr__(cfg.mongo, "uri", "mongodb://test")

    _setup_raw_csv(cfg, synthetic_df)

    inserted_batches = []
    fake_collection = MagicMock()
    fake_collection.insert_many.side_effect = lambda batch, ordered=False: MagicMock(
        inserted_ids=list(range(len(batch)))
    )

    @contextmanager
    def fake_mongo_collection(_mongo_cfg):
        inserted_batches.append("opened")
        yield fake_collection

    monkeypatch.setattr(ingest_module, "mongo_collection", fake_mongo_collection)

    result = ingest_module.run_ingest(cfg)

    assert result.pushed_to_mongo is True
    assert result.mongo_inserted == len(synthetic_df)
    fake_collection.drop.assert_called_once()
    assert fake_collection.insert_many.call_count >= 1


def test_run_ingest_raises_for_invalid_schema(tmp_path: Path, make_config, synthetic_df):
    cfg = make_config(tmp_path)
    bad = synthetic_df.drop(columns=["V1"])
    _setup_raw_csv(cfg, bad)

    try:
        ingest_module.run_ingest(cfg)
    except ValueError as exc:
        assert "schema validation" in str(exc)
    else:  # pragma: no cover - should never get here
        raise AssertionError("Expected ValueError for invalid schema")
