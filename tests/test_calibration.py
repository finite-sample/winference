"""Tests for calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from winference import brier_score, expected_calibration_error, log_loss, reliability_diagram
from winference.calibration import compare_calibration


class TestExpectedCalibrationError:
    def test_perfect_calibration_zero_ece(self) -> None:
        predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        observed = np.array([0, 0, 1, 1, 1])
        ece = expected_calibration_error(predicted, observed, n_bins=5)
        assert ece < 0.3

    def test_ece_in_range(self) -> None:
        predicted = np.random.rand(100)
        observed = np.random.randint(0, 2, 100).astype(float)
        ece = expected_calibration_error(predicted, observed)
        assert 0 <= ece <= 1

    def test_ece_zero_for_trivial_case(self) -> None:
        predicted = np.array([0.5, 0.5, 0.5, 0.5])
        observed = np.array([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(predicted, observed, n_bins=2)
        assert ece == pytest.approx(0.0, abs=0.01)


class TestBrierScore:
    def test_perfect_predictions_zero_brier(self) -> None:
        predicted = np.array([0.0, 0.0, 1.0, 1.0])
        observed = np.array([0, 0, 1, 1])
        bs = brier_score(predicted, observed)
        assert bs == pytest.approx(0.0)

    def test_worst_case_brier(self) -> None:
        predicted = np.array([1.0, 1.0, 0.0, 0.0])
        observed = np.array([0, 0, 1, 1])
        bs = brier_score(predicted, observed)
        assert bs == pytest.approx(1.0)

    def test_random_guessing_brier(self) -> None:
        predicted = np.full(1000, 0.5)
        observed = np.random.randint(0, 2, 1000).astype(float)
        bs = brier_score(predicted, observed)
        assert bs == pytest.approx(0.25, abs=0.05)


class TestLogLoss:
    def test_log_loss_positive(self) -> None:
        predicted = np.array([0.7, 0.3, 0.8])
        observed = np.array([1, 0, 1])
        ll = log_loss(predicted, observed)
        assert ll > 0

    def test_perfect_predictions_low_log_loss(self) -> None:
        predicted = np.array([0.999, 0.001, 0.999])
        observed = np.array([1, 0, 1])
        ll = log_loss(predicted, observed)
        assert ll < 0.01

    def test_log_loss_handles_clipping(self) -> None:
        predicted = np.array([0.0, 1.0])
        observed = np.array([0, 1])
        ll = log_loss(predicted, observed)
        assert np.isfinite(ll)


class TestReliabilityDiagram:
    def test_returns_dict_with_keys(self) -> None:
        predicted = np.random.rand(100)
        observed = np.random.randint(0, 2, 100).astype(float)
        result = reliability_diagram(predicted, observed)
        assert "bin_midpoints" in result
        assert "bin_accuracy" in result
        assert "bin_counts" in result
        assert "ece" in result

    def test_bin_midpoints_count(self) -> None:
        predicted = np.random.rand(100)
        observed = np.random.randint(0, 2, 100).astype(float)
        result = reliability_diagram(predicted, observed, n_bins=5)
        assert len(result["bin_midpoints"]) == 5

    def test_bin_counts_sum_to_n(self) -> None:
        n = 100
        predicted = np.random.rand(n)
        observed = np.random.randint(0, 2, n).astype(float)
        result = reliability_diagram(predicted, observed)
        assert result["bin_counts"].sum() == n


class TestCompareCalibration:
    def test_returns_dict_of_dicts(self) -> None:
        observed = np.array([1, 0, 1, 0, 1])
        methods = {
            "method_a": np.array([0.8, 0.2, 0.9, 0.1, 0.7]),
            "method_b": np.array([0.6, 0.4, 0.6, 0.4, 0.5]),
        }
        result = compare_calibration(methods, observed)
        assert "method_a" in result
        assert "method_b" in result
        assert "ece" in result["method_a"]
        assert "brier" in result["method_a"]
        assert "logloss" in result["method_a"]


class TestWithFixtures:
    def test_calibration_on_transitive_data(self, transitive_data: dict) -> None:
        from winference import BradleyTerry

        models = transitive_data["models"]
        comparisons = transitive_data["comparisons"]

        bt = BradleyTerry(models)
        bt.fit(comparisons)

        predicted = bt.predicted_win_rates(comparisons)
        observed = np.array([float(w) for _, _, w in comparisons])

        ece = expected_calibration_error(predicted, observed)
        bs = brier_score(predicted, observed)

        assert ece < 0.2
        assert bs < 0.3
