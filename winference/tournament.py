"""
Tournament graph diagnostics.

Build the directed tournament graph from pairwise outcomes and analyse its
strongly connected component (SCC) structure. Non-trivial SCCs (size > 1)
indicate clusters of models that cannot be linearly ordered — i.e. pockets
of non-transitivity.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class TournamentGraph:
    """Directed tournament graph built from pairwise comparison data.

    Args:
        models: Unique model identifiers.

    Examples:
        >>> tg = TournamentGraph(["A", "B", "C"])
        >>> tg.add_result("A", "B", win=True)
        >>> tg.add_result("B", "C", win=True)
        >>> tg.add_result("C", "A", win=True)
        >>> tg.scc_sizes()
        [3]
    """

    def __init__(self, models: list[str]) -> None:
        self.models = list(models)
        self._idx: dict[str, int] = {m: i for i, m in enumerate(self.models)}
        n = len(models)
        self.wins: NDArray[np.float64] = np.zeros((n, n), dtype=float)
        self.counts: NDArray[np.float64] = np.zeros((n, n), dtype=float)

    def add_result(self, model_a: str, model_b: str, win: bool) -> None:
        """Record a single comparison outcome (model_a wins if win=True)."""
        i, j = self._idx[model_a], self._idx[model_b]
        self.counts[i, j] += 1
        self.counts[j, i] += 1
        if win:
            self.wins[i, j] += 1
        else:
            self.wins[j, i] += 1

    def add_results_df(
        self,
        df: pd.DataFrame,
        col_a: str = "model_a",
        col_b: str = "model_b",
        col_win: str = "a_wins",
    ) -> None:
        """Bulk-load from a DataFrame with columns for model_a, model_b, a_wins (bool)."""
        for _, row in df.iterrows():
            self.add_result(str(row[col_a]), str(row[col_b]), bool(row[col_win]))

    def win_rate_matrix(self) -> NDArray[np.float64]:
        """Return the NxN matrix W where W[i,j] = P(i beats j)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where(self.counts > 0, self.wins / self.counts, 0.5)
        np.fill_diagonal(W, 0.5)
        return W

    def _build_adjacency(self, threshold: float = 0.5) -> dict[int, list[int]]:
        """Directed edge from i to j if win_rate(i,j) > threshold."""
        W = self.win_rate_matrix()
        adj: dict[int, list[int]] = {i: [] for i in range(len(self.models))}
        n = len(self.models)
        for i in range(n):
            for j in range(n):
                if i != j and W[i, j] > threshold:
                    adj[i].append(j)
        return adj

    def strongly_connected_components(self, threshold: float = 0.5) -> list[list[str]]:
        """Return SCCs via Tarjan's algorithm.

        Each SCC is a list of model names. SCCs of size > 1 represent
        non-transitive clusters.
        """
        adj = self._build_adjacency(threshold)
        n = len(self.models)
        index_counter = [0]
        stack: list[int] = []
        on_stack = [False] * n
        indices = [-1] * n
        lowlinks = [-1] * n
        sccs: list[list[int]] = []

        def strongconnect(v: int) -> None:
            indices[v] = lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True
            for w in adj[v]:
                if indices[w] == -1:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack[w]:
                    lowlinks[v] = min(lowlinks[v], indices[w])
            if lowlinks[v] == indices[v]:
                component: list[int] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                sccs.append(component)

        for v in range(n):
            if indices[v] == -1:
                strongconnect(v)

        return [[self.models[i] for i in scc] for scc in sccs]

    def scc_sizes(self, threshold: float = 0.5) -> list[int]:
        """Sorted list of SCC sizes (descending)."""
        sccs = self.strongly_connected_components(threshold)
        return sorted([len(s) for s in sccs], reverse=True)

    def nontransitivity_index(self, threshold: float = 0.5) -> float:
        """Fraction of models inside non-trivial SCCs (size > 1).

        0.0 = perfectly transitive, 1.0 = all models in one big cycle.
        """
        sccs = self.strongly_connected_components(threshold)
        n_in_cycles = sum(len(s) for s in sccs if len(s) > 1)
        return n_in_cycles / len(self.models)

    def count_cyclic_triples(self) -> tuple[int, int]:
        """Count cyclic vs total triples.

        Returns:
            Tuple of (n_cyclic, n_total). Under a transitive model n_cyclic = 0.
        """
        W = self.win_rate_matrix()
        n = len(self.models)
        n_cyclic = 0
        n_total = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    n_total += 1
                    if (W[i, j] > 0.5 and W[j, k] > 0.5 and W[k, i] > 0.5) or (
                        W[i, k] > 0.5 and W[k, j] > 0.5 and W[j, i] > 0.5
                    ):
                        n_cyclic += 1
        return n_cyclic, n_total

    def summary(self, threshold: float = 0.5) -> dict[str, int | float]:
        """Quick diagnostic summary."""
        sccs = self.strongly_connected_components(threshold)
        cyc, tot = self.count_cyclic_triples()
        return {
            "n_models": len(self.models),
            "n_sccs": len(sccs),
            "largest_scc": max(len(s) for s in sccs),
            "nontransitivity_index": self.nontransitivity_index(threshold),
            "cyclic_triples": cyc,
            "total_triples": tot,
            "cycle_fraction": cyc / tot if tot > 0 else 0.0,
        }
