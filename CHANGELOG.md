# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `GroupTest.group_labels`, a read-only view of the group label attached to
  each fitted comparison.

### Changed

- Adopted the py-canon fleet standard: shared reusable CI, docs and release
  workflows, ruff/pyright/pydoclint configuration, and a `src/` layout.

## 0.1.0 - 2026-03-08

### Added

- Initial release: Bradley-Terry fitting, Hodge decomposition of pairwise
  comparison matrices, heterogeneous group testing with per-group calibration,
  tournament-graph diagnostics, calibration metrics and simulation helpers.
