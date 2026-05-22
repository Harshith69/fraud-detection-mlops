# Fraud Detection MLOps Pipeline

End-to-end MLOps project for credit card fraud detection. The pipeline ingests data, persists it to MongoDB, snapshots it for DVC, trains and tunes Logistic Regression / XGBoost / LightGBM, and gates promotion to the MLflow Production stage on a measured F1 lift.

## Tech stack

- **Data versioning** — DVC + DagsHub remote
- **Long-term storage** — MongoDB Atlas (raw transactions)
- **Experiment tracking & registry** — MLflow on DagsHub
- **Modeling** — scikit-learn · XGBoost · LightGBM · Optuna
- **Orchestration** — DVC pipeline (`dvc.yaml`)
- **Tests** — pytest

## Project layout

```
src/fraud_detection/
  config.py            # loads params.yaml + .env
  logging_utils.py     # consistent logger
  data/
    schema.py          # column contract + validation
    mongo_client.py    # pymongo connection helper
    ingest.py          # CSV  -> Mongo + snapshot
    preprocess.py      # split + scale -> parquet
  models/
    factory.py         # build LR / XGB / LGBM
    mlflow_utils.py    # one-shot MLflow/DagsHub init
    tune.py            # Optuna + nested MLflow runs
    train.py           # final candidates, log + persist
    evaluate.py        # metric helpers
    register.py        # promotion gate vs Production
  pipeline/
    stages.py          # CLI surface for every stage
tests/                 # pytest suite (PBT-friendly)
params.yaml            # all tunables and paths
dvc.yaml               # ingest -> preprocess -> tune -> train -> register
```

## Setup

```cmd
:: 1. Create / activate a venv (Python 3.10+)
python -m venv fd_venv
fd_venv\Scripts\activate

:: 2. Install dependencies and the package in editable mode
pip install -r requirements.txt
pip install -e .

:: 3. Provide secrets in .env (copy from .env.example)
copy .env.example .env
:: then edit .env with your DagsHub token and MongoDB URI
```

## Running the pipeline

The pipeline is wired through DVC, so any change to data, code, or params triggers a rerun of the affected stages.

```cmd
:: Run any single stage
python -m fraud_detection.pipeline.stages ingest
python -m fraud_detection.pipeline.stages preprocess
python -m fraud_detection.pipeline.stages tune
python -m fraud_detection.pipeline.stages train
python -m fraud_detection.pipeline.stages register

:: Run all stages in order without DVC
python -m fraud_detection.pipeline.stages all

:: Or run the DVC pipeline (recommended)
dvc repro
```

## Promotion gate

When the data changes, `dvc repro` retrains and the **register** stage compares the new best candidate against the current Production version of the registered model on the F1 metric.

- If `candidate_f1 - production_f1 >= PROMOTION_MIN_DELTA`, the new version is registered, transitioned to Production, and the previous version is archived.
- Otherwise the incumbent stays in Production.

The decision and metrics are written to `reports/promotion.json`.

## Tests

```cmd
pytest
```

Tests cover schema validation, the preprocess transform, the model factory, evaluation helpers, ingest behavior with a mocked Mongo client, and the promotion decision matrix.

## Dataset

[Credit Card Fraud Detection 2023 (Kaggle)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023). The pipeline expects the CSV at `data/raw/creditcard_2023.csv`.
