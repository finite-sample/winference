"""Tests for HodgeDecomposition."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

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
        total = (
            result.transitive_variance
            + result.cyclic_variance
            + result.harmonic_variance
        )
        assert total == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize(
        ("counts", "cyclic_fraction"),
        [
            ([100, 1, 1], 50 / 1139),
            ([1, 1, 1], 1.0),
            ([0, 1, 1], 0.0),
            ([0, 0, 0], 0.0),
        ],
    )
    def test_weighted_triangle_fractions(
        self, counts: list[int], cyclic_fraction: float
    ) -> None:
        # The signed log odds for A>B, B>C, and C>A are all one. Residuals
        # must sum to three around the cycle. For positive counts their
        # minimum weighted squared length is 9 / sum(1 / count).
        rates = expit(np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]))
        ab, bc, ca = counts
        weights = np.array([[0, ab, ca], [ab, 0, bc], [ca, bc, 0]])
        fit = HodgeDecomposition(["A", "B", "C"]).fit(rates, weights)

        assert fit.cyclic_variance == pytest.approx(cyclic_fraction, abs=1e-12)
        transitive = 1 - cyclic_fraction if any(counts) else 0.0
        assert fit.transitive_variance == pytest.approx(transitive, abs=1e-12)
        assert fit.harmonic_variance == 0.0

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


class TestSparseCycles:
    def test_square_has_only_harmonic_flow(self) -> None:
        flow = np.zeros((4, 4))
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            flow[i, j], flow[j, i] = 1, -1
        result = HodgeDecomposition(list("ABCD")).fit(expit(flow), np.abs(flow))
        np.testing.assert_allclose(result.harmonic_flow, flow, atol=1e-12)
        np.testing.assert_allclose(result.curl_flow, 0, atol=1e-12)
        assert result.harmonic_variance == pytest.approx(1)

    @pytest.mark.parametrize(
        "counts", [np.ones((1, 2)), np.full((2, 2), -1), np.full((2, 2), np.nan)]
    )
    def test_invalid_counts(self, counts: np.ndarray) -> None:
        with pytest.raises(ValueError, match="count matrix"):
            HodgeDecomposition(list("AB")).fit(np.full((2, 2), 0.5), counts)

    def test_invalid_rate_shape(self) -> None:
        with pytest.raises(ValueError, match="win-rate matrix"):
            HodgeDecomposition(list("AB")).fit(np.full((1, 2), 0.5))

    def test_weighted_triangle_and_square(self) -> None:
        # The two cycles meet only at A. Each has balanced weighted flow at
        # every vertex, and only the triangle bounds a face.
        local = np.zeros((6, 6))
        global_cycle = np.zeros_like(local)
        counts = np.zeros_like(local)
        for flow, edges in [
            (local, [(0, 1, 4), (1, 2, 3), (2, 0, 2)]),
            (global_cycle, [(0, 3, 2), (3, 4, 5), (4, 5, 6), (5, 0, 3)]),
        ]:
            for i, j, w in edges:
                counts[i, j] = counts[j, i] = w
                flow[i, j], flow[j, i] = 1 / w, -1 / w
        potentials = [0, 0.05, 0.1, -0.3, 0.2, 0.4]
        rank = np.subtract.outer(potentials, potentials)
        observed = counts > 0
        rates = np.where(observed, expit(rank + local + global_cycle), np.nan)
        np.fill_diagonal(rates, 0.5)
        result = HodgeDecomposition(list("ABCDEF")).fit(rates, counts)
        for actual, expected in [
            (result.gradient_flow, rank),
            (result.curl_flow, local),
            (result.harmonic_flow, global_cycle),
        ]:
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        energies = np.array(
            [np.sum(counts * f**2) for f in (rank, local, global_cycle)]
        )
        np.testing.assert_allclose(
            [
                result.transitive_variance,
                result.cyclic_variance,
                result.harmonic_variance,
            ],
            energies / energies.sum(),
            atol=1e-12,
        )


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
        assert np.all(M >= 0)
        assert np.all(M <= 1)


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
