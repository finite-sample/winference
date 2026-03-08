"""
Bradley-Terry model fitting via maximum likelihood.

The BT model assumes P(i beats j) = sigma(theta_i - theta_j) where sigma
is the logistic function. Fitting recovers the strength vector theta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BradleyTerry:
    """Maximum-likelihood Bradley-Terry model.

    Args:
        models: Unique model identifiers.

    Examples:
        >>> bt = BradleyTerry(["A", "B", "C"])
        >>> bt.fit(comparisons)  # list of (i, j, outcome) tuples
        >>> bt.win_probability("A", "B")
        0.73
    """

    def __init__(self, models: list[str]) -> None:
        self.models = list(models)
        self._idx: dict[str, int] = {m: i for i, m in enumerate(self.models)}
        self.n = len(models)
        self.theta: NDArray[np.float64] | None = None
        self.loglik: float = 0.0
        self._fitted = False

    def fit(
        self,
        comparisons: list[tuple[str, str, bool]],
        reg: float = 1e-4,
    ) -> BradleyTerry:
        """Fit the model from a list of (model_a, model_b, a_wins) triples.

        Args:
            comparisons: Each element is (model_a, model_b, a_wins).
            reg: L2 regularisation strength (prevents divergence for unbeaten models).

        Returns:
            Self for method chaining.
        """
        data = np.array([(self._idx[a], self._idx[b], float(w)) for a, b, w in comparisons])
        idx_a = data[:, 0].astype(int)
        idx_b = data[:, 1].astype(int)
        outcomes = data[:, 2]

        def neg_loglik(theta: NDArray[np.float64]) -> float:
            logits = theta[idx_a] - theta[idx_b]
            ll = np.sum(outcomes * logits - np.logaddexp(0, logits))
            return float(-ll + 0.5 * reg * np.sum(theta**2))

        def grad(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            logits = theta[idx_a] - theta[idx_b]
            probs = expit(logits)
            residuals = outcomes - probs
            g = np.zeros(self.n)
            np.add.at(g, idx_a, residuals)
            np.add.at(g, idx_b, -residuals)
            return -(g - reg * theta)

        x0 = np.zeros(self.n)
        result = minimize(neg_loglik, x0, jac=grad, method="L-BFGS-B")
        self.theta = result.x - result.x.mean()
        self.loglik = -result.fun
        self._fitted = True
        return self

    def win_probability(self, model_a: str, model_b: str) -> float:
        """Predicted P(model_a beats model_b)."""
        if not self._fitted or self.theta is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        return float(expit(self.theta[self._idx[model_a]] - self.theta[self._idx[model_b]]))

    def win_probability_matrix(self) -> NDArray[np.float64]:
        """NxN matrix of predicted win probabilities."""
        if not self._fitted or self.theta is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        diff = self.theta[:, None] - self.theta[None, :]
        return expit(diff)

    def predicted_win_rates(
        self,
        comparisons: list[tuple[str, str, bool]],
    ) -> NDArray[np.float64]:
        """Return predicted P(a wins) for each comparison triple."""
        if not self._fitted:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        preds = np.array([self.win_probability(a, b) for a, b, _ in comparisons])
        return preds

    def strengths(self) -> dict[str, float]:
        """Return {model: theta} dictionary."""
        if not self._fitted or self.theta is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)
        return {m: float(self.theta[i]) for i, m in enumerate(self.models)}

    def rank(self) -> list[str]:
        """Models sorted by decreasing strength."""
        s = self.strengths()
        return sorted(s, key=lambda m: s[m], reverse=True)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"BradleyTerry(n_models={self.n}, {status})"


def fit_bt_from_matrix(
    win_matrix: NDArray[np.float64],
    count_matrix: NDArray[np.float64],
    models: list[str],
    reg: float = 1e-4,
) -> BradleyTerry:
    """Convenience: fit BT from win/count matrices rather than triples."""
    comparisons: list[tuple[str, str, bool]] = []
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            if count_matrix[i, j] > 0:
                w = round(win_matrix[i, j])
                losses = round(count_matrix[i, j] - win_matrix[i, j])
                comparisons.extend([(models[i], models[j], True)] * int(w))
                comparisons.extend([(models[i], models[j], False)] * int(losses))
    bt = BradleyTerry(models)
    bt.fit(comparisons, reg=reg)
    return bt
