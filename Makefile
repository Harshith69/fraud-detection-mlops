#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = fraud-detection-mlops
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies (and the local package in editable mode)
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Run a single pipeline stage. Usage: make stage NAME=ingest
.PHONY: stage
stage:
	$(PYTHON_INTERPRETER) -m fraud_detection.pipeline.stages $(NAME)

## Run the entire pipeline end-to-end
.PHONY: pipeline
pipeline:
	$(PYTHON_INTERPRETER) -m fraud_detection.pipeline.stages all

## Run the DVC pipeline (uses dvc.yaml stages)
.PHONY: dvc-repro
dvc-repro:
	dvc repro

## Serve the inference API locally with hot reload
.PHONY: api
api:
	$(PYTHON_INTERPRETER) -m uvicorn fraud_detection.api.app:app --reload --host 0.0.0.0 --port 8000

## Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t fraud-detection-mlops:dev .

## Run the API container locally
.PHONY: docker-run-api
docker-run-api:
	docker run --rm -p 8000:8000 --env-file .env fraud-detection-mlops:dev

## Run the pipeline container locally
.PHONY: docker-run-pipeline
docker-run-pipeline:
	docker run --rm --env-file .env -e APP_ROLE=pipeline fraud-detection-mlops:dev

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
