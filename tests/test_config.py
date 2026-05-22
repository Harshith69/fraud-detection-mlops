"""Tests for the configuration loader."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from fraud_detection import config as config_module


@pytest.fixture(autouse=True)
def _reset_config_cache():
    config_module.reload_config()
    yield
    config_module.reload_config()


def _write_params(path: Path, body: str) -> None:
    path.write_text(dedent(body), encoding="utf-8")


def test_get_config_resolves_paths_and_mongo_settings(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    params = project_root / "params.yaml"
    env = project_root / ".env"

    _write_params(
        params,
        """
        random_state: 7
        paths:
          raw_csv: data/raw/source.csv
        mlflow:
          experiment_name: test
          registered_model_name: m
          model_artifact_name: model
        register:
          promotion_metric: f1
          min_delta: 0.0
        """,
    )
    env.write_text(
        "MONGODB_URI=mongodb://example\nMONGODB_DB=mydb\nMONGODB_COLLECTION=mycol\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DB", raising=False)
    monkeypatch.delenv("MONGODB_COLLECTION", raising=False)

    cfg = config_module.get_config(params_file=params, env_file=env)

    assert cfg.params["random_state"] == 7
    assert cfg.mongo.uri == "mongodb://example"
    assert cfg.mongo.is_configured
    # `Config.path` resolves relative to the package PROJECT_ROOT, not the
    # test's tmp dir. Asserting that the relative segment is preserved is the
    # interesting part of the contract.
    resolved = cfg.path("raw_csv")
    assert resolved == (cfg.project_root / "data/raw/source.csv").resolve()


def test_section_raises_for_missing_key(tmp_path: Path):
    params = tmp_path / "params.yaml"
    _write_params(params, "random_state: 1\npaths: {}\n")
    cfg = config_module.get_config(params_file=params, env_file=tmp_path / ".no.env")

    with pytest.raises(KeyError):
        cfg.section("does_not_exist")


def test_path_raises_for_unknown_path(tmp_path: Path):
    params = tmp_path / "params.yaml"
    _write_params(params, "random_state: 1\npaths:\n  raw_csv: x.csv\n")
    cfg = config_module.get_config(params_file=params, env_file=tmp_path / ".no.env")

    with pytest.raises(KeyError):
        cfg.path("missing")


def test_strip_quotes_from_env(tmp_path: Path, monkeypatch):
    params = tmp_path / "params.yaml"
    env = tmp_path / ".env"
    _write_params(params, "random_state: 1\npaths: {}\n")
    env.write_text("DAGSHUB_USERNAME='quoted'\n", encoding="utf-8")
    monkeypatch.delenv("DAGSHUB_USERNAME", raising=False)

    cfg = config_module.get_config(params_file=params, env_file=env)
    assert cfg.dagshub.username == "quoted"
