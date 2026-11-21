# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.0 - 2026-11-20

### Added

- `GroupTest.group_labels`, a read-only view of the group label attached to
  each fitted comparison.

### Changed

- Adopted the py-canon fleet standard: shared reusable CI, docs and release
  workflows, ruff/pyright/pydoclint configuration, and a `src/` layout.

### Fixed

- Weight the Hodge variance fractions by the comparison counts used to fit
  the ranking. Pairs with zero count contribute zero to the reported fractions.
- Separate local triangle flow from global harmonic flow when some
  pairs have no comparisons. The sum of both variance fractions measures
  variation outside the fitted ranking.

## 0.1.0 - 2026-03-08

### Added

- Initial release: Bradley-Terry fitting, Hodge decomposition of pairwise
  comparison matrices, heterogeneous group testing with per-group calibration,
  tournament-graph diagnostics, calibration metrics and simulation helpers.
