#!/usr/bin/env sh
# Container entrypoint. Selects between the inference API and the pipeline
# runner via the APP_ROLE environment variable. Any positional arguments are
# forwarded to the chosen role.
set -eu

ROLE="${APP_ROLE:-api}"

case "$ROLE" in
    api)
        # Production-grade single-worker uvicorn. Increase workers in the
        # container runtime (e.g. via WEB_CONCURRENCY) if needed.
        exec uvicorn fraud_detection.api.app:app \
            --host 0.0.0.0 \
            --port "${PORT:-8000}" \
            "$@"
        ;;
    pipeline)
        STAGE="${PIPELINE_STAGE:-all}"
        exec python -m fraud_detection.pipeline.stages "$STAGE" "$@"
        ;;
    dvc-repro)
        # Pull DVC-tracked data + models and replay the pipeline.
        dvc pull || echo "dvc pull failed (non-fatal); continuing with local data"
        exec dvc repro "$@"
        ;;
    *)
        echo "Unknown APP_ROLE='$ROLE'. Expected 'api', 'pipeline', or 'dvc-repro'." >&2
        exit 64
        ;;
esac
