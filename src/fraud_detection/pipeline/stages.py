"""Single CLI surface for every pipeline stage.

Usage::

    python -m fraud_detection.pipeline.stages ingest
    python -m fraud_detection.pipeline.stages preprocess
    python -m fraud_detection.pipeline.stages tune
    python -m fraud_detection.pipeline.stages train
    python -m fraud_detection.pipeline.stages register
    python -m fraud_detection.pipeline.stages all

DVC stages call into these entry points so changing one stage's signature
only requires touching this file.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from ..data.ingest import run_ingest
from ..data.preprocess import run_preprocess
from ..logging_utils import get_logger
from ..models.register import run_register
from ..models.train import run_train
from ..models.tune import run_tune


_LOG = get_logger(__name__)


def _run_all() -> None:
    run_ingest()
    run_preprocess()
    run_tune()
    run_train()
    run_register()


STAGES: Dict[str, Callable[[], None]] = {
    "ingest": lambda: run_ingest(),
    "preprocess": lambda: run_preprocess(),
    "tune": lambda: run_tune(),
    "train": lambda: run_train(),
    "register": lambda: run_register(),
    "all": _run_all,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fraud detection pipeline runner")
    parser.add_argument("stage", choices=sorted(STAGES.keys()), help="Stage to run")
    args = parser.parse_args(argv)
    _LOG.info("Running stage: %s", args.stage)
    STAGES[args.stage]()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
