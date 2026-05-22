# Docker

A single image serves two roles selected by the `APP_ROLE` environment variable:

| `APP_ROLE`  | Behaviour                                                                |
|-------------|--------------------------------------------------------------------------|
| `api`       | Launches uvicorn on `:8000` serving `fraud_detection.api.app:app`        |
| `pipeline`  | Runs `python -m fraud_detection.pipeline.stages all` and exits          |
| `dvc-repro` | Runs `dvc pull && dvc repro` (used by the retrain workflow)              |

## Build

```bash
docker build -t fraud-detection:dev .
```

## Run as inference API

```bash
docker run --rm -p 8000:8000 \
  -e DAGSHUB_USERNAME=... \
  -e DAGSHUB_REPO=fraud-detection-mlops \
  -e DAGSHUB_TOKEN=... \
  fraud-detection:dev
```

Then `GET http://localhost:8000/health` and `POST /predict`.

## Run as pipeline (one-shot)

```bash
docker run --rm \
  -e APP_ROLE=pipeline \
  -e DAGSHUB_USERNAME=... \
  -e DAGSHUB_REPO=fraud-detection-mlops \
  -e DAGSHUB_TOKEN=... \
  -e MONGODB_URI=... \
  -v %cd%/data:/app/data \
  -v %cd%/models:/app/models \
  fraud-detection:dev
```

The image deliberately does not bake in `data/raw/*.csv` or trained models —
those are pulled by DVC at runtime, or mounted in by the operator.
