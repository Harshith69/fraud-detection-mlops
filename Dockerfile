# syntax=docker/dockerfile:1.7

############################
# Stage 1: builder
############################
# Install dependencies into a virtualenv we can copy into the runtime image.
# Using --mount=type=cache keeps repeat builds fast in CI.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# libgomp1 is required by xgboost / lightgbm even at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip install --no-deps .

############################
# Stage 2: runtime
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ROLE=api

# libgomp1 is needed at runtime by xgboost / lightgbm.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src ./src
COPY --chown=app:app params.yaml ./params.yaml
COPY --chown=app:app dvc.yaml ./dvc.yaml
COPY --chown=app:app docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/data /app/models /app/reports /app/mlruns \
    && chown -R app:app /app

USER app
EXPOSE 8000

# APP_ROLE=api    -> launch FastAPI via uvicorn (default)
# APP_ROLE=pipeline -> run the full pipeline once and exit
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
