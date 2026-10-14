# Changelog

All notable changes to BanditLab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-10-14

### Added
- Top-level API: `from mab_framework import ThompsonSampling, ...`
- CLI entry points: `bandit run/benchmark/evaluate`, `bandit-gui`
- `python -m mab_framework` support
- Optional extras: `[deep]`, `[optimization]`, `[obp]`, `[gui]`, `[viz]`, `[all]`
- Built-in benchmarks now ship with the wheel
- 10 runnable examples in `examples/`
- `importlib.metadata`-based `__version__`
- `tqdm` as a declared dependency
- `docs/` directory scaffold

### Changed
- `BaseAlgorithm` now normalizes a single model into a per-arm list (deep-copied)
- `NeuralUCBAlgorithm` rewritten as a per-arm variant with per-arm `Z` matrices
- Wheel size reduced from 28 MB to ~4 MB (excluded `.idea`, `data`, `plots`, `scripts`, `tests`)
- README updated for the new API and CLI

### Fixed
- `pip install banditlab` no longer pulls PyTorch / cvxpy / coba by default
- `bandit-gui` now prints a friendly message when `streamlit` is missing
- `bandit run` works with `SyntheticLinearEnv` configs out of the box

## [1.0.0] — 2026-09-08

### Added
- Initial public release.
