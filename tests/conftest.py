"""Pytest fixtures using simulators."""

from __future__ import annotations

import pytest

from winference.simulate import (
    simulate_heterogeneous,
    simulate_llm_arena,
    simulate_rock_paper_scissors,
    simulate_transitive,
)


@pytest.fixture
def transitive_data() -> dict:
    """Pure BT data with no cycles."""
    return simulate_transitive(n_models=5, n_comparisons=2000, seed=42)


@pytest.fixture
def heterogeneous_data() -> dict:
    """Data with per-category strengths."""
    return simulate_heterogeneous(
        n_models=5,
        n_categories=3,
        n_comparisons=3000,
        seed=42,
    )


@pytest.fixture
def cyclic_data() -> dict:
    """Data with irreducible cyclic structure."""
    return simulate_rock_paper_scissors(
        n_models=5,
        n_comparisons=3000,
        cycle_strength=0.8,
        seed=42,
    )


@pytest.fixture
def llm_arena_data() -> dict:
    """Realistic LLM arena scenario."""
    return simulate_llm_arena(seed=42)
