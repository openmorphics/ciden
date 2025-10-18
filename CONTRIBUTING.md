# Contributing to SR-CIDEN

Thanks for your interest in contributing!

## Quick start (development)

- Create a virtual environment (or conda) and install with extras:
  - `pip install -e .[dev,docs]`
- Install pre-commit hooks:
  - `pre-commit install`
- Run the test suite and docs build:
  - `pytest -q`
  - `mkdocs build`

## Style and quality

- Formatting: `black` (88 chars), imports: `isort` (black profile), lint: `ruff`
- Type checking: `mypy` (Python 3.10 target)
- Notebooks: outputs are stripped via `nbstripout` in pre-commit
- License headers: Apache-2.0 headers are enforced by pre-commit

## Reproducibility

- Determinism helpers: see [src/sr_ciden/utils/determinism.py](src/sr_ciden/utils/determinism.py:1)
- Run the one-shot paper artifacts script:
  - `scripts/reproduce_paper.sh`

## Issues and pull requests

- Open issues at https://github.com/openmorphics/ciden/issues
- For PRs, please:
  - Reference related issues
  - Include tests for new functionality
  - Update docs where applicable
