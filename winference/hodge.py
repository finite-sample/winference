"""
Hodge decomposition of pairwise comparison data.

Any skew-symmetric matrix of log-odds (or edge flows on the tournament
graph) can be orthogonally decomposed into:

    Y = grad(s) + curl(C) + harmonic

where:
  - grad(s) is the *transitive* component: there exists a potential s_i
    for each node such that Y_ij ~ s_i - s_j.  This is the part that can
    be calibrated via a standard Bradley-Terry model.
  - curl(C) is the *cyclic* component: it captures rock-paper-scissors
    structure that is irreducible to any linear ranking.
  - harmonic captures global topological structure (zero for complete
    tournaments).

The decomposition is computed via least-squares on the combinatorial
Laplacian, following Jiang et al. (2011) "Statistical ranking and
combinatorial Hodge theory".

Calibration strategy
--------------------
1. Compute the Hodge decomposition of the empirical log-odds.
2. Use the gradient component s as BT strengths giving *transitive*
   calibrated win probabilities: P_trans(i > j) = sigmoid(s_i - s_j).
3. Report ||curl||^2 / ||Y||^2 as the fraction of variance due to
   non-transitive structure, the part your calibration *ignores*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.special import expit, logit

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HodgeResult(NamedTuple):
    """Result of a Hodge decomposition."""

    potential: NDArray[np.float64]
    gradient_flow: NDArray[np.float64]
    curl_flow: NDArray[np.float64]
    harmonic_flow: NDArray[np.float64]
    transitive_variance: float
    cyclic_variance: float
    harmonic_variance: float


class HodgeDecomposition:
    """Hodge decomposition of a pairwise comparison matrix.

    Args:
        models: Model identifiers.

    Examples:
        >>> hd = HodgeDecomposition(["A", "B", "C", "D"])
        >>> result = hd.fit(win_rate_matrix)
        >>> print(f"Cyclic fraction: {result.cyclic_variance:.1%}")
        >>> # Calibrate using only the transitive part
        >>> p_trans = hd.transitive_win_probability("A", "B")
    """

    def __init__(self, models: list[str]) -> None:
        self.models = list(models)
        self._idx: dict[str, int] = {m: i for i, m in enumerate(self.models)}
        self.n = len(models)
        self.result: HodgeResult | None = None

    def fit(
        self,
        W: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
    ) -> HodgeResult:
        """Decompose a win-rate matrix.

        Args:
            W: Win rate matrix of shape (n, n) where W[i,j] = P(i beats j).
                Should satisfy W[i,j] + W[j,i] ~ 1.
            weights: Number of comparisons per pair of shape (n, n), used for
                weighted least squares. Default: uniform weights.

        Returns:
            HodgeResult containing decomposition components.
        """
        n = self.n

        W_clip = np.clip(W, 0.01, 0.99)
        Y = logit(W_clip)
        Y = 0.5 * (Y - Y.T)
        np.fill_diagonal(Y, 0.0)

        if weights is None:
            weights = np.ones((n, n))
        weights = 0.5 * (weights + weights.T)

        L = np.zeros((n, n))
        div_Y = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    w = weights[i, j]
                    L[i, i] += w
                    L[i, j] -= w
                    div_Y[i] += w * Y[i, j]

        L_red = L[1:, 1:]
        div_red = div_Y[1:]
        s_red = np.linalg.lstsq(L_red, div_red, rcond=None)[0]
        s = np.zeros(n)
        s[1:] = s_red
        s -= s.mean()

        grad_flow = np.subtract.outer(s, s)
        curl_flow = Y - grad_flow
        harmonic_flow = np.zeros_like(Y)

        def sq_norm(M: NDArray[np.float64]) -> float:
            return float(np.sum(np.triu(M, k=1) ** 2))

        total_var = sq_norm(Y)
        grad_var = sq_norm(grad_flow)
        curl_var = sq_norm(curl_flow)
        harm_var = sq_norm(harmonic_flow)

        if total_var > 0:
            tv = grad_var / total_var
            cv = curl_var / total_var
            hv = harm_var / total_var
        else:
            tv = cv = hv = 0.0

        self.result = HodgeResult(
            potential=s,
            gradient_flow=grad_flow,
            curl_flow=curl_flow,
            harmonic_flow=harmonic_flow,
            transitive_variance=tv,
            cyclic_variance=cv,
            harmonic_variance=hv,
        )
        return self.result

    def transitive_win_probability(self, model_a: str, model_b: str) -> float:
        """P(a beats b) using only the gradient (transitive) component.

        This is the win probability that *can* be calibrated to a scalar
        ranking.  The cyclic component is dropped.
        """
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        s = self.result.potential
        return float(expit(s[self._idx[model_a]] - s[self._idx[model_b]]))

    def transitive_win_matrix(self) -> NDArray[np.float64]:
        """Full NxN transitive win probability matrix."""
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        s = self.result.potential
        return expit(np.subtract.outer(s, s))

    def transitive_strengths(self) -> dict[str, float]:
        """Hodge potential (transitive strength) per model."""
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        return {m: float(self.result.potential[i]) for i, m in enumerate(self.models)}

    def curl_magnitude_per_pair(self) -> NDArray[np.float64]:
        """NxN matrix of curl magnitude: how much each pair deviates from transitivity."""
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        return np.abs(self.result.curl_flow)

    def worst_pairs(self, k: int = 10) -> list[tuple[str, str, float]]:
        """Top-k pairs with largest cyclic residual."""
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        C = np.abs(self.result.curl_flow)
        pairs: list[tuple[str, str, float]] = []
        n = self.n
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.models[i], self.models[j], float(C[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:k]

    def summary(self) -> dict[str, float | int]:
        """Quick summary of the decomposition."""
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        r = self.result
        return {
            "transitive_variance_frac": r.transitive_variance,
            "cyclic_variance_frac": r.cyclic_variance,
            "harmonic_variance_frac": r.harmonic_variance,
            "n_models": self.n,
        }
