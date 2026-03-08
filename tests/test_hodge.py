"""Tests for HodgeDecomposition."""

from __future__ import annotations

import numpy as np
import pytest

from winference import HodgeDecomposition, TournamentGraph


class TestHodgeDecompositionInit:
    def test_init_stores_models(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        assert hd.models == ["A", "B", "C"]
        assert hd.n == 3

    def test_init_result_none(self) -> None:
        hd = HodgeDecomposition(["A", "B"])
        assert hd.result is None


class TestFit:
    def test_returns_hodge_result(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        result = hd.fit(W)
        assert result is not None
        assert hd.result is result

    def test_result_has_all_fields(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        result = hd.fit(W)
        assert hasattr(result, "potential")
        assert hasattr(result, "gradient_flow")
        assert hasattr(result, "curl_flow")
        assert hasattr(result, "harmonic_flow")
        assert hasattr(result, "transitive_variance")
        assert hasattr(result, "cyclic_variance")

    def test_potential_centered(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C", "D"])
        W = np.array(
            [
                [0.5, 0.7, 0.8, 0.9],
                [0.3, 0.5, 0.6, 0.7],
                [0.2, 0.4, 0.5, 0.6],
                [0.1, 0.3, 0.4, 0.5],
            ]
        )
        result = hd.fit(W)
        assert abs(result.potential.mean()) < 1e-10


class TestVarianceFractions:
    def test_variance_fractions_sum_to_one(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C", "D"])
        W = np.array(
            [
                [0.5, 0.6, 0.7, 0.4],
                [0.4, 0.5, 0.8, 0.3],
                [0.3, 0.2, 0.5, 0.6],
                [0.6, 0.7, 0.4, 0.5],
            ]
        )
        result = hd.fit(W)
        total = result.transitive_variance + result.cyclic_variance + result.harmonic_variance
        assert total == pytest.approx(1.0, abs=0.01)

    def test_transitive_data_high_transitive_variance(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C", "D"])
        W = np.array(
            [
                [0.5, 0.9, 0.95, 0.99],
                [0.1, 0.5, 0.9, 0.95],
                [0.05, 0.1, 0.5, 0.9],
                [0.01, 0.05, 0.1, 0.5],
            ]
        )
        result = hd.fit(W)
        assert result.transitive_variance > 0.9


class TestTransitiveWinProbability:
    def test_raises_before_fit(self) -> None:
        hd = HodgeDecomposition(["A", "B"])
        with pytest.raises(RuntimeError):
            hd.transitive_win_probability("A", "B")

    def test_probability_in_range(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        p = hd.transitive_win_probability("A", "B")
        assert 0 <= p <= 1

    def test_symmetry(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        p_ab = hd.transitive_win_probability("A", "B")
        p_ba = hd.transitive_win_probability("B", "A")
        assert p_ab + p_ba == pytest.approx(1.0)


class TestTransitiveWinMatrix:
    def test_shape(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        M = hd.transitive_win_matrix()
        assert M.shape == (3, 3)

    def test_values_in_range(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        M = hd.transitive_win_matrix()
        assert np.all(M >= 0) and np.all(M <= 1)


class TestTransitiveStrengths:
    def test_returns_dict(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        s = hd.transitive_strengths()
        assert isinstance(s, dict)
        assert "A" in s


class TestCurlMagnitude:
    def test_returns_array(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        C = hd.curl_magnitude_per_pair()
        assert C.shape == (3, 3)
        assert np.all(C >= 0)


class TestWorstPairs:
    def test_returns_list(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        pairs = hd.worst_pairs(k=2)
        assert isinstance(pairs, list)
        assert len(pairs) <= 2

    def test_sorted_descending(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C", "D"])
        W = np.array(
            [
                [0.5, 0.6, 0.7, 0.4],
                [0.4, 0.5, 0.8, 0.3],
                [0.3, 0.2, 0.5, 0.6],
                [0.6, 0.7, 0.4, 0.5],
            ]
        )
        hd.fit(W)
        pairs = hd.worst_pairs(k=5)
        magnitudes = [p[2] for p in pairs]
        assert magnitudes == sorted(magnitudes, reverse=True)


class TestSummary:
    def test_summary_keys(self) -> None:
        hd = HodgeDecomposition(["A", "B", "C"])
        W = np.array([[0.5, 0.7, 0.8], [0.3, 0.5, 0.6], [0.2, 0.4, 0.5]])
        hd.fit(W)
        summary = hd.summary()
        assert "transitive_variance_frac" in summary
        assert "cyclic_variance_frac" in summary
        assert "n_models" in summary


class TestWithFixtures:
    def test_cyclic_data_has_nonzero_curl(self, cyclic_data: dict) -> None:
        models = cyclic_data["models"]
        comparisons = cyclic_data["comparisons"]

        tg = TournamentGraph(models)
        for a, b, w in comparisons:
            tg.add_result(a, b, w)
        W = tg.win_rate_matrix()

        hd = HodgeDecomposition(models)
        result = hd.fit(W)
        assert result.cyclic_variance > 0.01

    def test_transitive_data_low_curl(self, transitive_data: dict) -> None:
        models = transitive_data["models"]
        comparisons = transitive_data["comparisons"]

        tg = TournamentGraph(models)
        for a, b, w in comparisons:
            tg.add_result(a, b, w)
        W = tg.win_rate_matrix()

        hd = HodgeDecomposition(models)
        result = hd.fit(W)
        assert result.transitive_variance > 0.8
