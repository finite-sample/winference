"""Hodge decomposition of pairwise comparison data.

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
3. Add the cyclic and harmonic variance fractions to measure the share
   of comparison-weighted variation outside a single ranking.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.sparse.linalg import lsmr
from scipy.special import expit, logit


class HodgeResult(NamedTuple):
    """Result of a Hodge decomposition."""

    potential: NDArray[np.float64]
    gradient_flow: NDArray[np.float64]
    curl_flow: NDArray[np.float64]
    harmonic_flow: NDArray[np.float64]
    transitive_variance: float
    cyclic_variance: float
    harmonic_variance: float


def _split_residual(
    residual: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project the observed residual onto the boundaries of observed triangles."""
    observed = weights > 0
    np.fill_diagonal(observed, False)
    residual = np.where(observed, residual, 0.0)
    # A vertex joined to every other vertex fills every cycle with triangles,
    # including on complete tournaments. No harmonic component can remain.
    if np.any(observed.sum(axis=1) == len(weights) - 1):
        return residual, np.zeros_like(residual)

    left, right = np.nonzero(np.triu(observed, k=1))
    edge_index = np.zeros_like(weights, dtype=np.intp)
    edge_index[left, right] = np.arange(len(left))
    triangle_edges = []
    for i, j in zip(left, right, strict=True):
        later = np.flatnonzero(observed[i, j + 1 :] & observed[j, j + 1 :]) + j + 1
        triangle_edges.extend(
            (edge_index[i, j], edge_index[j, k], edge_index[i, k]) for k in later
        )
    if not triangle_edges:
        return np.zeros_like(residual), residual

    # In coordinates scaled by sqrt(weight), a triangle contributes signed
    # 1/sqrt(weight) to each of its three edges. Its column space is the
    # local component of the weighted Hodge decomposition (Jiang et al. §5.1).
    root_weight = np.sqrt(weights[left, right])
    rows = np.asarray(triangle_edges, dtype=np.intp).ravel()
    cols = np.repeat(np.arange(len(triangle_edges)), 3)
    signs = np.tile([1.0, 1.0, -1.0], len(triangle_edges))
    boundaries = coo_array(
        (signs / root_weight[rows], (rows, cols)),
        shape=(len(left), len(triangle_edges)),
    ).tocsr()
    target = root_weight * residual[left, right]
    coefficients, stop, *_ = lsmr(
        boundaries,
        target,
        atol=1e-13,
        btol=1e-13,
        conlim=0,
        maxiter=5 * min(len(left), len(triangle_edges)),
    )
    if stop in (3, 6, 7):
        raise RuntimeError("The weighted triangle projection did not converge")
    local = np.zeros_like(residual)
    local[left, right] = (boundaries @ coefficients) / root_weight
    local[right, left] = -local[left, right]
    return local, residual - local


class HodgeDecomposition:
    """Hodge decomposition of a pairwise comparison matrix.

    Examples:
        >>> import numpy as np
        >>> from winference import HodgeDecomposition
        >>> rps = np.array([[0.5, 0.9, 0.1], [0.1, 0.5, 0.9], [0.9, 0.1, 0.5]])
        >>> hd = HodgeDecomposition(["A", "B", "C"])
        >>> result = hd.fit(rps)
        >>> result.cyclic_variance  # rock-paper-scissors is pure curl
        1.0
        >>> hd.transitive_win_probability("A", "B")  # no transitive signal left
        0.5
    """

    def __init__(self, models: list[str]) -> None:
        """Create an unfitted decomposition over ``models``.

        Args:
            models: Model names, in the order used for indexing.
        """
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
                Observed pairs should satisfy W[i,j] + W[j,i] ~ 1. Values
                at pairs with zero weight are ignored.
            weights: Number of comparisons per pair of shape (n, n), used for
                weighted least squares and the reported variance fractions.
                A pair with zero comparisons contributes no weight. By default
                every pair has equal weight.

        Returns:
            HodgeResult containing decomposition components. The gradient gives
            fitted differences for all model pairs. Local curl and harmonic
            flow are zero on unobserved pairs.

        Raises:
            ValueError: For invalid matrix shapes or comparison counts.
        """
        n = self.n

        if weights is None:
            weights = np.ones((n, n))
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n, n) or not np.all(np.isfinite(weights) & (weights >= 0)):
            raise ValueError(
                "weights must be a finite, nonnegative (n, n) count matrix"
            )
        weights = 0.5 * weights + 0.5 * weights.T
        if np.shape(W) != (n, n):
            raise ValueError("W must be an (n, n) win-rate matrix")
        W_clip = np.clip(np.where(weights > 0, W, 0.5), 0.01, 0.99)
        Y = logit(W_clip)
        Y = 0.5 * (Y - Y.T)
        np.fill_diagonal(Y, 0.0)

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
        curl_flow, harmonic_flow = _split_residual(Y - grad_flow, weights)

        # The projection is orthogonal in the same weighted inner product
        # used in the least-squares fit (Jiang et al., 2011, section 5.1).
        edge_weights = np.triu(weights, k=1)

        def sq_norm(M: NDArray[np.float64]) -> float:
            return float(np.sum(edge_weights * M**2))

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
        """Absolute local triangle flow per pair."""
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        return np.abs(self.result.curl_flow)

    def worst_pairs(self, k: int = 10) -> list[tuple[str, str, float]]:
        """Top-k pairs with the largest absolute local triangle flow."""
        if self.result is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        C = np.abs(self.result.curl_flow)
        n = self.n
        pairs: list[tuple[str, str, float]] = [
            (self.models[i], self.models[j], float(C[i, j]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
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
