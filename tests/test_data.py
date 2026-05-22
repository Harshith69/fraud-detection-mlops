"""Smoke test that the package imports cleanly."""

import importlib


def test_package_imports():
    module = importlib.import_module("fraud_detection")
    assert hasattr(module, "__version__")
