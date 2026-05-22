"""Central configuration loader.

Configuration is layered in this order (later sources override earlier):

1. ``params.yaml`` — pipeline parameters that are versioned alongside the code.
2. ``.env`` — local secrets and per-environment settings. Optional; only loaded
   if the file exists. Never committed.
3. Process environment — values already in ``os.environ`` always win. This is
   how GitHub Actions / Docker / Kubernetes inject secrets without a ``.env``.

The :class:`Config` dataclass exposes typed access so the rest of the codebase
never reaches into ``os.environ`` or parses YAML directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import yaml

# Project layout. Every path in :class:`Config` resolves against PROJECT_ROOT
# so the package works no matter what CWD a stage runs from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_FILE = PROJECT_ROOT / "params.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class MongoConfig:
    """MongoDB connection settings sourced from the environment."""

    uri: str
    database: str
    collection: str

    @property
    def is_configured(self) -> bool:
        return bool(self.uri and self.database and self.collection)


@dataclass(frozen=True)
class DagsHubConfig:
    """DagsHub credentials used by both MLflow and DVC."""

    username: str
    repo: str
    token: str

    @property
    def is_configured(self) -> bool:
        return bool(self.username and self.repo)

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"https://dagshub.com/{self.username}/{self.repo}.mlflow"


@dataclass(frozen=True)
class Config:
    """Resolved configuration for a single pipeline invocation."""

    project_root: Path
    params: Dict[str, Any]
    mongo: MongoConfig
    dagshub: DagsHubConfig
    random_state: int
    promotion_metric: str
    promotion_min_delta: float

    # Convenience helpers ----------------------------------------------------

    def path(self, key: str) -> Path:
        """Resolve a path declared under ``params.paths`` against the project root."""
        try:
            relative = self.params["paths"][key]
        except KeyError as exc:
            raise KeyError(f"Path '{key}' is not declared in params.yaml under 'paths'.") from exc
        return (self.project_root / relative).resolve()

    def section(self, name: str) -> Dict[str, Any]:
        """Return a top-level params section, raising a clear error if absent."""
        try:
            return self.params[name]
        except KeyError as exc:
            raise KeyError(f"Params section '{name}' is missing from params.yaml.") from exc


def _load_params(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"params.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"params.yaml must define a mapping at the top level, got {type(loaded)!r}"
        )
    return loaded


def _strip(value: str | None) -> str:
    """Remove surrounding quotes/whitespace from .env values."""
    if value is None:
        return ""
    return value.strip().strip("'").strip('"')


@lru_cache(maxsize=1)
def get_config(
    params_file: Path | None = None,
    env_file: Path | None = None,
) -> Config:
    """Load and cache the project configuration.

    Subsequent calls return the same :class:`Config` instance unless
    :func:`reload_config` is used. Tests can pass explicit paths to bypass
    caching by calling :func:`reload_config` first.
    """

    params_path = params_file or DEFAULT_PARAMS_FILE
    env_path = env_file or DEFAULT_ENV_FILE

    # `.env` is local-only and optional. Process env vars (e.g. GitHub Actions
    # secrets, Kubernetes secrets) are read directly from os.environ and take
    # precedence over `.env` because we pass override=False.
    if env_path.exists():
        load_dotenv(env_path, override=False)

    params = _load_params(params_path)

    mongo = MongoConfig(
        uri=_strip(os.getenv("MONGODB_URI")),
        database=_strip(os.getenv("MONGODB_DB")) or "fraud_detection",
        collection=_strip(os.getenv("MONGODB_COLLECTION")) or "creditcard_transactions",
    )

    dagshub = DagsHubConfig(
        username=_strip(os.getenv("DAGSHUB_USERNAME")),
        repo=_strip(os.getenv("DAGSHUB_REPO")),
        token=_strip(os.getenv("DAGSHUB_TOKEN")),
    )

    random_state = int(_strip(os.getenv("RANDOM_STATE")) or params.get("random_state", 42))
    promotion_metric = _strip(os.getenv("PROMOTION_METRIC")) or params.get("register", {}).get(
        "promotion_metric", "f1"
    )
    promotion_min_delta = float(
        _strip(os.getenv("PROMOTION_MIN_DELTA"))
        or params.get("register", {}).get("min_delta", 0.0)
    )

    return Config(
        project_root=PROJECT_ROOT,
        params=params,
        mongo=mongo,
        dagshub=dagshub,
        random_state=random_state,
        promotion_metric=promotion_metric,
        promotion_min_delta=promotion_min_delta,
    )


def reload_config() -> None:
    """Clear the cached :class:`Config`. Useful in tests when env or params change."""
    get_config.cache_clear()
