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
