"""Tests for TournamentGraph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winference import TournamentGraph


class TestTournamentGraphInit:
    def test_init_creates_zero_matrices(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        assert tg.wins.shape == (3, 3)
        assert tg.counts.shape == (3, 3)
        assert np.all(tg.wins == 0)
        assert np.all(tg.counts == 0)

    def test_models_stored(self) -> None:
        models = ["X", "Y", "Z"]
        tg = TournamentGraph(models)
        assert tg.models == models


class TestAddResult:
    def test_single_win(self) -> None:
        tg = TournamentGraph(["A", "B"])
        tg.add_result("A", "B", win=True)
        assert tg.wins[0, 1] == 1
        assert tg.wins[1, 0] == 0
        assert tg.counts[0, 1] == 1
        assert tg.counts[1, 0] == 1

    def test_single_loss(self) -> None:
        tg = TournamentGraph(["A", "B"])
        tg.add_result("A", "B", win=False)
        assert tg.wins[0, 1] == 0
        assert tg.wins[1, 0] == 1

    def test_multiple_results(self) -> None:
        tg = TournamentGraph(["A", "B"])
        for _ in range(10):
            tg.add_result("A", "B", win=True)
        for _ in range(5):
            tg.add_result("A", "B", win=False)
        assert tg.wins[0, 1] == 10
        assert tg.wins[1, 0] == 5
        assert tg.counts[0, 1] == 15


class TestAddResultsDf:
    def test_bulk_load(self) -> None:
        df = pd.DataFrame(
            {
                "model_a": ["A", "A", "B"],
                "model_b": ["B", "B", "A"],
                "a_wins": [True, False, True],
            }
        )
        tg = TournamentGraph(["A", "B"])
        tg.add_results_df(df)
        assert tg.counts[0, 1] == 3


class TestWinRateMatrix:
    def test_win_rate_computation(self) -> None:
        tg = TournamentGraph(["A", "B"])
        for _ in range(7):
            tg.add_result("A", "B", win=True)
        for _ in range(3):
            tg.add_result("A", "B", win=False)
        W = tg.win_rate_matrix()
        assert W[0, 1] == pytest.approx(0.7)
        assert W[1, 0] == pytest.approx(0.3)

    def test_diagonal_is_half(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        W = tg.win_rate_matrix()
        assert W[0, 0] == 0.5
        assert W[1, 1] == 0.5
        assert W[2, 2] == 0.5

    def test_no_data_defaults_to_half(self) -> None:
        tg = TournamentGraph(["A", "B"])
        W = tg.win_rate_matrix()
        assert W[0, 1] == 0.5


class TestSCC:
    def test_transitive_single_sccs(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        tg.add_result("B", "C", win=True)
        tg.add_result("A", "C", win=True)
        sccs = tg.strongly_connected_components()
        assert all(len(scc) == 1 for scc in sccs)

    def test_cycle_gives_single_large_scc(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        tg.add_result("B", "C", win=True)
        tg.add_result("C", "A", win=True)
        sccs = tg.strongly_connected_components()
        sizes = [len(s) for s in sccs]
        assert 3 in sizes

    def test_scc_sizes_sorted_descending(self) -> None:
        tg = TournamentGraph(["A", "B", "C", "D"])
        tg.add_result("A", "B", win=True)
        tg.add_result("B", "C", win=True)
        tg.add_result("C", "A", win=True)
        tg.add_result("A", "D", win=True)
        sizes = tg.scc_sizes()
        assert sizes == sorted(sizes, reverse=True)


class TestNontransitivityIndex:
    def test_transitive_gives_zero(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        tg.add_result("B", "C", win=True)
        tg.add_result("A", "C", win=True)
        assert tg.nontransitivity_index() == 0.0

    def test_full_cycle_gives_one(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        tg.add_result("B", "C", win=True)
        tg.add_result("C", "A", win=True)
        assert tg.nontransitivity_index() == 1.0


class TestCyclicTriples:
    def test_no_cycles_in_transitive(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        tg.add_result("B", "C", win=True)
        tg.add_result("A", "C", win=True)
        n_cyclic, n_total = tg.count_cyclic_triples()
        assert n_cyclic == 0
        assert n_total == 1

    def test_one_cycle_detected(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        tg.add_result("B", "C", win=True)
        tg.add_result("C", "A", win=True)
        n_cyclic, n_total = tg.count_cyclic_triples()
        assert n_cyclic == 1
        assert n_total == 1


class TestSummary:
    def test_summary_keys(self) -> None:
        tg = TournamentGraph(["A", "B", "C"])
        tg.add_result("A", "B", win=True)
        summary = tg.summary()
        assert "n_models" in summary
        assert "n_sccs" in summary
        assert "largest_scc" in summary
        assert "nontransitivity_index" in summary
        assert "cyclic_triples" in summary
        assert "cycle_fraction" in summary


class TestWithFixtures:
    def test_transitive_data_low_nontransitivity(self, transitive_data: dict) -> None:
        models = transitive_data["models"]
        comparisons = transitive_data["comparisons"]
        tg = TournamentGraph(models)
        for a, b, w in comparisons:
            tg.add_result(a, b, w)
        assert tg.nontransitivity_index() < 0.5

    def test_cyclic_data_has_nonzero_curl(self, cyclic_data: dict) -> None:
        from winference import HodgeDecomposition

        models = cyclic_data["models"]
        comparisons = cyclic_data["comparisons"]
        tg = TournamentGraph(models)
        for a, b, w in comparisons:
            tg.add_result(a, b, w)
        hd = HodgeDecomposition(models)
        result = hd.fit(tg.win_rate_matrix())
        assert result.cyclic_variance > 0.01
