"""Tests for BradleyTerry model."""

from __future__ import annotations

import numpy as np
import pytest

from winference import BradleyTerry
from winference.bradley_terry import fit_bt_from_matrix


class TestBradleyTerryInit:
    def test_init_stores_models(self) -> None:
        bt = BradleyTerry(["A", "B", "C"])
        assert bt.models == ["A", "B", "C"]
        assert bt.n == 3

    def test_init_not_fitted(self) -> None:
        bt = BradleyTerry(["A", "B"])
        assert not bt._fitted
        assert bt.theta is None


class TestFit:
    def test_fit_returns_self(self) -> None:
        bt = BradleyTerry(["A", "B"])
        comparisons = [("A", "B", True), ("A", "B", False)]
        result = bt.fit(comparisons)
        assert result is bt

    def test_fit_sets_fitted_flag(self) -> None:
        bt = BradleyTerry(["A", "B"])
        bt.fit([("A", "B", True)])
        assert bt._fitted

    def test_theta_centered_at_zero(self) -> None:
        bt = BradleyTerry(["A", "B", "C"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 10
        bt.fit(comparisons)
        assert bt.theta is not None
        assert abs(bt.theta.mean()) < 0.01

    def test_stronger_model_has_higher_theta(self) -> None:
        bt = BradleyTerry(["A", "B"])
        comparisons = [("A", "B", True)] * 20 + [("A", "B", False)] * 5
        bt.fit(comparisons)
        assert bt.theta is not None
        assert bt.theta[0] > bt.theta[1]


class TestWinProbability:
    def test_raises_before_fit(self) -> None:
        bt = BradleyTerry(["A", "B"])
        with pytest.raises(RuntimeError):
            bt.win_probability("A", "B")

    def test_probability_in_range(self) -> None:
        bt = BradleyTerry(["A", "B"])
        bt.fit([("A", "B", True)] * 10)
        p = bt.win_probability("A", "B")
        assert 0 <= p <= 1

    def test_symmetry(self) -> None:
        bt = BradleyTerry(["A", "B"])
        bt.fit([("A", "B", True)] * 10 + [("A", "B", False)] * 5)
        p_ab = bt.win_probability("A", "B")
        p_ba = bt.win_probability("B", "A")
        assert p_ab + p_ba == pytest.approx(1.0)

    def test_equal_strength_gives_half(self) -> None:
        bt = BradleyTerry(["A", "B"])
        comparisons = [("A", "B", True)] * 50 + [("A", "B", False)] * 50
        bt.fit(comparisons)
        p = bt.win_probability("A", "B")
        assert p == pytest.approx(0.5, abs=0.1)


class TestWinProbabilityMatrix:
    def test_shape(self) -> None:
        bt = BradleyTerry(["A", "B", "C"])
        bt.fit([("A", "B", True), ("B", "C", True)] * 10)
        W = bt.win_probability_matrix()
        assert W.shape == (3, 3)

    def test_diagonal_is_half(self) -> None:
        bt = BradleyTerry(["A", "B"])
        bt.fit([("A", "B", True)] * 10)
        W = bt.win_probability_matrix()
        assert W[0, 0] == pytest.approx(0.5)
        assert W[1, 1] == pytest.approx(0.5)


class TestStrengths:
    def test_returns_dict(self) -> None:
        bt = BradleyTerry(["A", "B"])
        bt.fit([("A", "B", True)] * 10)
        s = bt.strengths()
        assert isinstance(s, dict)
        assert "A" in s and "B" in s


class TestRank:
    def test_rank_order(self) -> None:
        bt = BradleyTerry(["A", "B", "C"])
        comparisons = [("A", "B", True), ("B", "C", True), ("A", "C", True)] * 20
        bt.fit(comparisons)
        ranking = bt.rank()
        assert ranking[0] == "A"
        assert ranking[-1] == "C"


class TestPredictedWinRates:
    def test_returns_array(self) -> None:
        bt = BradleyTerry(["A", "B"])
        comparisons = [("A", "B", True)] * 10
        bt.fit(comparisons)
        preds = bt.predicted_win_rates(comparisons)
        assert len(preds) == 10
        assert all(0 <= p <= 1 for p in preds)


class TestRepr:
    def test_repr_unfitted(self) -> None:
        bt = BradleyTerry(["A", "B"])
        assert "unfitted" in repr(bt)

    def test_repr_fitted(self) -> None:
        bt = BradleyTerry(["A", "B"])
        bt.fit([("A", "B", True)])
        assert "fitted" in repr(bt)


class TestFitBtFromMatrix:
    def test_fit_from_matrix(self) -> None:
        models = ["A", "B"]
        win_matrix = np.array([[0, 8], [2, 0]], dtype=float)
        count_matrix = np.array([[0, 10], [10, 0]], dtype=float)
        bt = fit_bt_from_matrix(win_matrix, count_matrix, models)
        assert bt._fitted
        assert bt.win_probability("A", "B") > 0.5


class TestWithFixtures:
    def test_recovers_ranking_from_transitive(self, transitive_data: dict) -> None:
        models = transitive_data["models"]
        comparisons = transitive_data["comparisons"]
        true_strengths = transitive_data["true_strengths"]

        bt = BradleyTerry(models)
        bt.fit(comparisons)

        true_ranking = sorted(models, key=lambda m: true_strengths[m], reverse=True)
        fitted_ranking = bt.rank()

        top_match = fitted_ranking[0] == true_ranking[0]
        bottom_match = fitted_ranking[-1] == true_ranking[-1]
        assert top_match or bottom_match
