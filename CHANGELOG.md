# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Packaging metadata finalization in [pyproject.toml](pyproject.toml:1): python_requires (>=3.10), project.urls (Documentation, Paper, Repository, Issues), extended classifiers including Python 3.12, long_description content-type.
- Documentation site bootstrap with MkDocs Material and mkdocstrings: [mkdocs.yml](mkdocs.yml:1), [docs/index.md](docs/index.md:1).
- Citation and metadata files: [CITATION.cff](CITATION.cff:1), [codemeta.json](codemeta.json:1).
- Developer quality tooling: [.pre-commit-config.yaml](.pre-commit-config.yaml:1), [.editorconfig](.editorconfig:1), license header template [LICENSE-HEADER](LICENSE-HEADER:1).
- README docs badge and link to GitHub Pages: [README.md](README.md:1).

### Changed
- Adopt Ruff + Black + isort + mypy configuration under [pyproject.toml](pyproject.toml:45).

### Security
- Pre-commit hooks include Bandit and pip-audit (manual stage) to support CI gates.

## [0.1.0] - 2025-10-16
### Added
- Initial repository structure with `src/` layout, examples, benchmarks, and minimal tests.
- Basic packaging skeleton with [pyproject.toml](pyproject.toml:1).
- README quickstart and minimal training/inference example [examples/minimal_training_loop.py](examples/minimal_training_loop.py:1).

[Unreleased]: https://github.com/openmorphics/ciden/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/openmorphics/ciden/releases/tag/v0.1.0
