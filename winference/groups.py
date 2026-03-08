"""
Heterogeneous group testing and per-group calibration.

Tests whether model strengths are constant across prompt categories
(homogeneous BT) or differ by category (heterogeneous BT).  If
heterogeneity is warranted, fits per-category BT models and provides
composable win rate predictions for any target distribution over categories.

The formal test is a likelihood-ratio test:
    H0: theta_{i,k} = theta_i  for all i, k   (homogeneous)
    H1: theta_{i,k} free                       (heterogeneous)

    Lambda = -2(l0 - l1)  ~  chi2  with (K-1)(N-1) degrees of freedom.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from winference.bradley_terry import BradleyTerry

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GroupTest:
    """Likelihood-ratio test for heterogeneity across prompt groups.

    Args:
        models: Model identifiers.
        groups: Unique group/category labels.

    Examples:
        >>> gt = GroupTest(models=["A","B","C"], groups=["math","creative"])
        >>> gt.fit(comparisons, group_labels)
        >>> print(gt.test_result())
        {'statistic': 14.2, 'df': 2, 'p_value': 0.0008}
    """

    def __init__(self, models: list[str], groups: list[str]) -> None:
        self.models = list(models)
        self.groups = list(groups)
        self.n_models = len(models)
        self.n_groups = len(groups)
        self.bt_null: BradleyTerry | None = None
        self.bt_per_group: dict[str, BradleyTerry] = {}
        self._comparisons: list[tuple[str, str, bool]] = []
        self._group_labels: list[str] = []
        self._group_comparisons: dict[str, list[tuple[str, str, bool]]] = {}

    def fit(
        self,
        comparisons: list[tuple[str, str, bool]],
        group_labels: list[str],
        reg: float = 1e-4,
    ) -> GroupTest:
        """Fit null (pooled) and alternative (per-group) BT models.

        Args:
            comparisons: List of (model_a, model_b, a_wins) tuples.
            group_labels: Category label for each comparison, same length as comparisons.
            reg: Regularisation for BT fitting.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If comparisons and group_labels have different lengths.
        """
        if len(comparisons) != len(group_labels):
            msg = "comparisons and group_labels must have same length"
            raise ValueError(msg)

        self.bt_null = BradleyTerry(self.models)
        self.bt_null.fit(comparisons, reg=reg)

        self.bt_per_group = {}
        self._group_comparisons = {}
        for g in self.groups:
            g_comps = [c for c, gl in zip(comparisons, group_labels, strict=True) if gl == g]
            if len(g_comps) < self.n_models:
                continue
            self._group_comparisons[g] = g_comps
            bt_g = BradleyTerry(self.models)
            bt_g.fit(g_comps, reg=reg)
            self.bt_per_group[g] = bt_g

        self._comparisons = comparisons
        self._group_labels = group_labels
        return self

    def _loglik(self, bt: BradleyTerry, comparisons: list[tuple[str, str, bool]]) -> float:
        """Compute log-likelihood of data under a fitted BT model."""
        ll = 0.0
        for a, b, w in comparisons:
            p = bt.win_probability(a, b)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            ll += np.log(p) if w else np.log(1 - p)
        return float(ll)

    def test_result(self) -> dict[str, float | int | bool]:
        """Likelihood-ratio test for group heterogeneity.

        Returns:
            Dict with keys: statistic (LRT statistic Lambda), df (degrees of freedom),
            p_value (p-value under chi2 null), reject_at_05 (bool).

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.bt_null is None:
            msg = "Call .fit() first"
            raise RuntimeError(msg)

        ll_null = self._loglik(self.bt_null, self._comparisons)

        ll_alt = 0.0
        for g, bt_g in self.bt_per_group.items():
            ll_alt += self._loglik(bt_g, self._group_comparisons[g])
        for g in self.groups:
            if g not in self.bt_per_group:
                g_comps = [
                    c
                    for c, gl in zip(self._comparisons, self._group_labels, strict=True)
                    if gl == g
                ]
                ll_alt += self._loglik(self.bt_null, g_comps)

        lrt = -2 * (ll_null - ll_alt)
        K = len(self.bt_per_group)
        N = self.n_models
        df = (K - 1) * (N - 1)

        if df <= 0:
            return {"statistic": 0.0, "df": 0, "p_value": 1.0, "reject_at_05": False}

        p_value = float(1 - stats.chi2.cdf(lrt, df))
        return {
            "statistic": float(lrt),
            "df": df,
            "p_value": p_value,
            "reject_at_05": p_value < 0.05,
        }

    def per_group_strengths(self) -> dict[str, dict[str, float]]:
        """Return {group: {model: theta}} for each fitted group."""
        return {g: bt.strengths() for g, bt in self.bt_per_group.items()}


class GroupCalibrator:
    """Composable win rate calibration using per-group BT models.

    After fitting per-group BT models, compute win rates for *any*
    target distribution over groups:

        P(i > j | pi*) = sum_k  pi*_k * sigmoid(theta_{i,k} - theta_{j,k})

    This is the key advantage: calibration that transfers under
    distribution shift.

    Args:
        group_test: A fitted GroupTest object.

    Raises:
        RuntimeError: If the GroupTest has not been fitted.
    """

    def __init__(self, group_test: GroupTest) -> None:
        if group_test.bt_null is None:
            msg = "GroupTest must be fitted"
            raise RuntimeError(msg)
        self.gt = group_test

    def win_probability(
        self,
        model_a: str,
        model_b: str,
        target_distribution: dict[str, float] | None = None,
    ) -> float:
        """Composite win probability under a target group distribution.

        Args:
            model_a: First model name.
            model_b: Second model name.
            target_distribution: Dict mapping group to weight. Weights are
                normalised internally. If None, uses the empirical distribution
                from the training data.

        Returns:
            Composite win probability P(model_a beats model_b).
        """
        if target_distribution is None:
            target_distribution = self._empirical_distribution()

        total = sum(target_distribution.values())
        pi = {g: w / total for g, w in target_distribution.items()}

        p = 0.0
        for g, weight in pi.items():
            if g in self.gt.bt_per_group:
                p += weight * self.gt.bt_per_group[g].win_probability(model_a, model_b)
            elif self.gt.bt_null is not None:
                p += weight * self.gt.bt_null.win_probability(model_a, model_b)
        return p

    def win_probability_matrix(
        self,
        target_distribution: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        """NxN composite win probability matrix."""
        models = self.gt.models
        n = len(models)
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    W[i, j] = self.win_probability(models[i], models[j], target_distribution)
                else:
                    W[i, j] = 0.5
        return W

    def _empirical_distribution(self) -> dict[str, float]:
        """Group weights proportional to number of comparisons."""
        counts = Counter(self.gt._group_labels)
        return dict(counts)

    def sensitivity_analysis(
        self,
        model_a: str,
        model_b: str,
        n_draws: int = 1000,
        concentration: float = 1.0,
    ) -> dict[str, float]:
        """How much does P(a > b) vary as the target distribution changes?

        Draws random target distributions from Dirichlet(concentration)
        and reports the range and std of the composite win probability.
        """
        groups = list(self.gt.bt_per_group.keys())
        K = len(groups)
        if K < 2:
            p = self.win_probability(model_a, model_b)
            return {"mean": p, "std": 0.0, "min": p, "max": p}

        alpha = np.full(K, concentration)
        draws = np.random.dirichlet(alpha, size=n_draws)

        probs = []
        for draw in draws:
            pi = dict(zip(groups, draw, strict=True))
            probs.append(self.win_probability(model_a, model_b, pi))
        probs_arr = np.array(probs)

        return {
            "mean": float(probs_arr.mean()),
            "std": float(probs_arr.std()),
            "min": float(probs_arr.min()),
            "max": float(probs_arr.max()),
            "q05": float(np.quantile(probs_arr, 0.05)),
            "q95": float(np.quantile(probs_arr, 0.95)),
        }
