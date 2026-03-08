"""Tests for GroupTest and GroupCalibrator."""

from __future__ import annotations

import pytest

from winference import GroupCalibrator, GroupTest


class TestGroupTestInit:
    def test_init_stores_models_and_groups(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["math", "coding"])
        assert gt.models == ["A", "B"]
        assert gt.groups == ["math", "coding"]
        assert gt.n_models == 2
        assert gt.n_groups == 2

    def test_init_not_fitted(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["x"])
        assert gt.bt_null is None


class TestGroupTestFit:
    def test_fit_returns_self(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["x"])
        comparisons = [("A", "B", True), ("A", "B", False)]
        group_labels = ["x", "x"]
        result = gt.fit(comparisons, group_labels)
        assert result is gt

    def test_fit_sets_null_model(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["x"])
        comparisons = [("A", "B", True)] * 10
        group_labels = ["x"] * 10
        gt.fit(comparisons, group_labels)
        assert gt.bt_null is not None
        assert gt.bt_null._fitted

    def test_fit_creates_per_group_models(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 5
        group_labels = ["x", "x", "x", "y", "y", "y"] * 2 + ["x", "x", "x"]
        gt.fit(comparisons, group_labels)
        assert len(gt.bt_per_group) > 0

    def test_raises_on_length_mismatch(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["x"])
        with pytest.raises(ValueError):
            gt.fit([("A", "B", True)], ["x", "y"])


class TestGroupTestResult:
    def test_raises_before_fit(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["x"])
        with pytest.raises(RuntimeError):
            gt.test_result()

    def test_result_has_required_keys(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 15 + ["y"] * 15
        gt.fit(comparisons, group_labels)
        result = gt.test_result()
        assert "statistic" in result
        assert "df" in result
        assert "p_value" in result
        assert "reject_at_05" in result

    def test_statistic_nonnegative(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 15 + ["y"] * 15
        gt.fit(comparisons, group_labels)
        result = gt.test_result()
        assert result["statistic"] >= -0.01

    def test_p_value_in_range(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 15 + ["y"] * 15
        gt.fit(comparisons, group_labels)
        result = gt.test_result()
        assert 0 <= result["p_value"] <= 1


class TestPerGroupStrengths:
    def test_returns_dict(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 15 + ["y"] * 15
        gt.fit(comparisons, group_labels)
        strengths = gt.per_group_strengths()
        assert isinstance(strengths, dict)


class TestGroupCalibratorInit:
    def test_raises_on_unfitted_group_test(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["x"])
        with pytest.raises(RuntimeError):
            GroupCalibrator(gt)

    def test_init_with_fitted_group_test(self) -> None:
        gt = GroupTest(models=["A", "B"], groups=["x"])
        gt.fit([("A", "B", True)] * 10, ["x"] * 10)
        gc = GroupCalibrator(gt)
        assert gc.gt is gt


class TestGroupCalibratorWinProbability:
    def test_probability_in_range(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 15 + ["y"] * 15
        gt.fit(comparisons, group_labels)
        gc = GroupCalibrator(gt)
        p = gc.win_probability("A", "B")
        assert 0 <= p <= 1

    def test_custom_distribution(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 15 + ["y"] * 15
        gt.fit(comparisons, group_labels)
        gc = GroupCalibrator(gt)
        p = gc.win_probability("A", "B", target_distribution={"x": 0.8, "y": 0.2})
        assert 0 <= p <= 1


class TestGroupCalibratorWinProbabilityMatrix:
    def test_shape(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 30
        gt.fit(comparisons, group_labels)
        gc = GroupCalibrator(gt)
        W = gc.win_probability_matrix()
        assert W.shape == (3, 3)


class TestSensitivityAnalysis:
    def test_returns_dict_with_stats(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x", "y"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 15 + ["y"] * 15
        gt.fit(comparisons, group_labels)
        gc = GroupCalibrator(gt)
        result = gc.sensitivity_analysis("A", "B", n_draws=100)
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result

    def test_single_group_no_variation(self) -> None:
        gt = GroupTest(models=["A", "B", "C"], groups=["x"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        group_labels = ["x"] * 30
        gt.fit(comparisons, group_labels)
        gc = GroupCalibrator(gt)
        result = gc.sensitivity_analysis("A", "B")
        assert result["std"] == 0.0


class TestWithFixtures:
    def test_heterogeneous_data_rejects_null(self, heterogeneous_data: dict) -> None:
        models = heterogeneous_data["models"]
        comparisons = heterogeneous_data["comparisons"]
        categories = heterogeneous_data["categories"]
        groups = sorted(set(categories))

        gt = GroupTest(models, groups)
        gt.fit(comparisons, categories)
        result = gt.test_result()

        assert result["statistic"] > 0

    def test_llm_arena_heterogeneity(self, llm_arena_data: dict) -> None:
        models = llm_arena_data["models"]
        comparisons = llm_arena_data["comparisons"]
        categories = llm_arena_data["categories"]
        groups = sorted(set(categories))

        gt = GroupTest(models, groups)
        gt.fit(comparisons, categories)
        result = gt.test_result()

        assert result["reject_at_05"]
