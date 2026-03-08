"""Tests for simulation functions."""

from __future__ import annotations

import pytest

from winference.simulate import (
    simulate_heterogeneous,
    simulate_llm_arena,
    simulate_rock_paper_scissors,
    simulate_transitive,
)


class TestSimulateTransitive:
    def test_returns_dict_with_required_keys(self) -> None:
        data = simulate_transitive()
        assert "comparisons" in data
        assert "models" in data
        assert "true_strengths" in data
        assert "categories" in data

    def test_correct_number_of_models(self) -> None:
        data = simulate_transitive(n_models=4)
        assert len(data["models"]) == 4
        assert len(data["true_strengths"]) == 4

    def test_correct_number_of_comparisons(self) -> None:
        data = simulate_transitive(n_comparisons=100)
        assert len(data["comparisons"]) == 100
        assert len(data["categories"]) == 100

    def test_comparisons_structure(self) -> None:
        data = simulate_transitive(n_models=3, n_comparisons=10)
        for comp in data["comparisons"]:
            assert len(comp) == 3
            assert comp[0] in data["models"]
            assert comp[1] in data["models"]
            assert isinstance(comp[2], bool)

    def test_reproducibility(self) -> None:
        data1 = simulate_transitive(seed=123)
        data2 = simulate_transitive(seed=123)
        assert data1["comparisons"] == data2["comparisons"]
        assert data1["true_strengths"] == data2["true_strengths"]

    def test_different_seeds_different_results(self) -> None:
        data1 = simulate_transitive(seed=1)
        data2 = simulate_transitive(seed=2)
        assert data1["true_strengths"] != data2["true_strengths"]

    def test_categories_all_general(self) -> None:
        data = simulate_transitive()
        assert all(c == "general" for c in data["categories"])


class TestSimulateHeterogeneous:
    def test_returns_dict_with_required_keys(self) -> None:
        data = simulate_heterogeneous()
        assert "comparisons" in data
        assert "models" in data
        assert "true_strengths" in data
        assert "categories" in data
        assert "category_weights" in data

    def test_correct_number_of_categories(self) -> None:
        data = simulate_heterogeneous(n_categories=4)
        assert len(data["true_strengths"]) == 4
        assert len(data["category_weights"]) == 4

    def test_category_weights_sum_to_one(self) -> None:
        data = simulate_heterogeneous()
        total = sum(data["category_weights"].values())
        assert total == pytest.approx(1.0)

    def test_custom_category_names(self) -> None:
        names = ["math", "coding", "writing"]
        data = simulate_heterogeneous(n_categories=3, category_names=names)
        assert set(data["true_strengths"].keys()) == set(names)

    def test_raises_on_wrong_category_names_length(self) -> None:
        with pytest.raises(ValueError):
            simulate_heterogeneous(n_categories=3, category_names=["a", "b"])

    def test_reproducibility(self) -> None:
        data1 = simulate_heterogeneous(seed=42)
        data2 = simulate_heterogeneous(seed=42)
        assert data1["comparisons"] == data2["comparisons"]

    def test_per_category_strengths_structure(self) -> None:
        data = simulate_heterogeneous(n_models=4, n_categories=2)
        for cat_strengths in data["true_strengths"].values():
            assert len(cat_strengths) == 4


class TestSimulateRockPaperScissors:
    def test_returns_dict_with_required_keys(self) -> None:
        data = simulate_rock_paper_scissors()
        assert "comparisons" in data
        assert "models" in data
        assert "true_transitive" in data
        assert "true_curl_magnitude" in data
        assert "categories" in data

    def test_curl_magnitude_in_range(self) -> None:
        data = simulate_rock_paper_scissors()
        assert 0 <= data["true_curl_magnitude"] <= 1

    def test_higher_cycle_strength_more_curl(self) -> None:
        data_low = simulate_rock_paper_scissors(cycle_strength=0.2)
        data_high = simulate_rock_paper_scissors(cycle_strength=1.5)
        assert data_high["true_curl_magnitude"] > data_low["true_curl_magnitude"]

    def test_reproducibility(self) -> None:
        data1 = simulate_rock_paper_scissors(seed=99)
        data2 = simulate_rock_paper_scissors(seed=99)
        assert data1["comparisons"] == data2["comparisons"]

    def test_categories_all_general(self) -> None:
        data = simulate_rock_paper_scissors()
        assert all(c == "general" for c in data["categories"])


class TestSimulateLlmArena:
    def test_returns_dict_with_required_keys(self) -> None:
        data = simulate_llm_arena()
        assert "comparisons" in data
        assert "models" in data
        assert "true_strengths" in data
        assert "categories" in data
        assert "category_weights" in data

    def test_six_models(self) -> None:
        data = simulate_llm_arena()
        assert len(data["models"]) == 6

    def test_three_categories(self) -> None:
        data = simulate_llm_arena()
        assert len(data["true_strengths"]) == 3

    def test_model_names(self) -> None:
        data = simulate_llm_arena()
        expected = ["AlphaLM", "BetaChat", "GammaCoder", "DeltaWrite", "EpsilonAll", "ZetaMath"]
        assert set(data["models"]) == set(expected)

    def test_category_names(self) -> None:
        data = simulate_llm_arena()
        expected = {"reasoning", "creative_writing", "coding"}
        assert set(data["true_strengths"].keys()) == expected

    def test_reproducibility(self) -> None:
        data1 = simulate_llm_arena(seed=42)
        data2 = simulate_llm_arena(seed=42)
        assert data1["comparisons"] == data2["comparisons"]

    def test_has_8000_comparisons(self) -> None:
        data = simulate_llm_arena()
        assert len(data["comparisons"]) == 8000
