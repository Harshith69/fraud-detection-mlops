"""Stage 2: split the snapshot into train/test parquet files and fit the scaler.

Inputs:
    - ``params.paths.raw_snapshot`` (CSV)

Outputs (DVC-tracked):
    - ``params.paths.train_features`` / ``train_labels``
    - ``params.paths.test_features`` / ``test_labels``
    - ``params.paths.scaler``

Only the ``Amount`` column is scaled. The PCA features V1..V28 are already
zero-mean / unit-variance in this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import Config, get_config
from ..logging_utils import get_logger
from .schema import ID_COLUMN, TARGET_COLUMN, feature_columns, validate_dataframe


_LOG = get_logger(__name__)


@dataclass
class PreprocessResult:
    """Summary returned by :func:`run_preprocess`."""

    train_rows: int
    test_rows: int
    train_features_path: Path
    train_labels_path: Path
    test_features_path: Path
    test_labels_path: Path
    scaler_path: Path


def split_and_scale(
    df: pd.DataFrame,
    *,
    test_size: float,
    stratify: bool,
    random_state: int,
):
    """Pure-Python helper that performs the train/test split and scaling.

    Returns ``(X_train_scaled, X_test_scaled, y_train, y_test, scaler)``.
    Kept separate from :func:`run_preprocess` so unit tests can exercise the
    transformation without touching disk.
    """

    df_model = df.drop(columns=[ID_COLUMN]) if ID_COLUMN in df.columns else df.copy()
    features = feature_columns(df_model)

    X = df_model[features]
    y = df_model[TARGET_COLUMN].astype(int)

    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    if "Amount" in features:
        X_train_scaled["Amount"] = scaler.fit_transform(X_train[["Amount"]])
        X_test_scaled["Amount"] = scaler.transform(X_test[["Amount"]])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def run_preprocess(cfg: Config | None = None) -> PreprocessResult:
    """Read the snapshot, split, scale, and persist outputs."""

    cfg = cfg or get_config()
    snapshot = cfg.path("raw_snapshot")
    pre = cfg.section("preprocess")

    if not snapshot.exists():
        raise FileNotFoundError(
            f"Snapshot CSV not found at {snapshot}. Run the ingest stage first."
        )

    _LOG.info("Reading snapshot from %s", snapshot)
    df = pd.read_csv(snapshot)

    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise ValueError(
            "Snapshot failed schema validation: " + "; ".join(validation.errors)
        )

    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        df,
        test_size=float(pre.get("test_size", 0.2)),
        stratify=bool(pre.get("stratify", True)),
        random_state=cfg.random_state,
    )

    processed_dir = cfg.path("processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_features_path = cfg.path("train_features")
    train_labels_path = cfg.path("train_labels")
    test_features_path = cfg.path("test_features")
    test_labels_path = cfg.path("test_labels")
    scaler_path = cfg.path("scaler")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(train_features_path, index=False)
    X_test.to_parquet(test_features_path, index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_parquet(train_labels_path, index=False)
    y_test.to_frame(name=TARGET_COLUMN).to_parquet(test_labels_path, index=False)
    joblib.dump(scaler, scaler_path)

    _LOG.info(
        "Preprocess wrote train=%d test=%d to %s",
        len(X_train),
        len(X_test),
        processed_dir,
    )

    return PreprocessResult(
        train_rows=len(X_train),
        test_rows=len(X_test),
        train_features_path=train_features_path,
        train_labels_path=train_labels_path,
        test_features_path=test_features_path,
        test_labels_path=test_labels_path,
        scaler_path=scaler_path,
    )


def load_processed(cfg: Config):
    """Load the parquet train/test split written by :func:`run_preprocess`."""
    X_train = pd.read_parquet(cfg.path("train_features"))
    X_test = pd.read_parquet(cfg.path("test_features"))
    y_train = pd.read_parquet(cfg.path("train_labels"))[TARGET_COLUMN].astype(int)
    y_test = pd.read_parquet(cfg.path("test_labels"))[TARGET_COLUMN].astype(int)
    return X_train, X_test, y_train, y_test


def main() -> None:
    """CLI entry point used by the DVC stage."""
    result = run_preprocess()
    _LOG.info(
        "Preprocess complete: train=%d test=%d", result.train_rows, result.test_rows
    )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
