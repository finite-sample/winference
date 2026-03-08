"""
Simulate pairwise comparison data with controlled non-transitivity.

Three generators corresponding to different stories about where
non-transitivity comes from:

1. ``simulate_transitive`` — pure BT data (no cycles).
2. ``simulate_heterogeneous`` — models have different strengths per
   category.  Aggregate win rates can cycle even though within-category
   preferences are transitive.
3. ``simulate_rock_paper_scissors`` — irreducible cyclic structure that
   cannot be explained by category decomposition.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import expit

if TYPE_CHECKING:
    from numpy.typing import NDArray


def simulate_transitive(
    n_models: int = 6,
    n_comparisons: int = 3000,
    strength_spread: float = 1.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate data from a standard BT model (fully transitive).

    Args:
        n_models: Number of models to simulate.
        n_comparisons: Number of pairwise comparisons to generate.
        strength_spread: Standard deviation of model strengths.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: comparisons (list of (model_a, model_b, a_wins)),
        models (list of str), true_strengths (dict {model: theta}),
        categories (list of str, all "general").
    """
    rng = np.random.default_rng(seed)
    models = [f"M{i}" for i in range(n_models)]
    theta: NDArray[np.float64] = rng.normal(0, strength_spread, size=n_models)
    theta = theta - theta.mean()

    comparisons: list[tuple[str, str, bool]] = []
    categories: list[str] = []
    for _ in range(n_comparisons):
        i, j = rng.choice(n_models, 2, replace=False)
        p = expit(theta[i] - theta[j])
        win = rng.random() < p
        comparisons.append((models[i], models[j], bool(win)))
        categories.append("general")

    return {
        "comparisons": comparisons,
        "models": models,
        "true_strengths": {m: float(theta[k]) for k, m in enumerate(models)},
        "categories": categories,
    }


def simulate_heterogeneous(
    n_models: int = 6,
    n_categories: int = 3,
    n_comparisons: int = 5000,
    strength_spread: float = 1.5,
    category_names: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate data where model strengths differ by category.

    Each model has a different strength in each category.  Within each
    category, preferences are transitive.  But the aggregate (marginal
    over categories) can exhibit non-transitive win rates.

    Args:
        n_models: Number of models to simulate.
        n_categories: Number of categories.
        n_comparisons: Number of pairwise comparisons to generate.
        strength_spread: Standard deviation of model strengths.
        category_names: Optional custom category names.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: comparisons (list of (model_a, model_b, a_wins)),
        models (list of str), true_strengths (dict {category: {model: theta}}),
        categories (list of str, one per comparison),
        category_weights (dict {category: proportion}).

    Raises:
        ValueError: If category_names length doesn't match n_categories.
    """
    rng = np.random.default_rng(seed)
    models = [f"M{i}" for i in range(n_models)]

    if category_names is None:
        category_names = [f"cat_{k}" for k in range(n_categories)]
    if len(category_names) != n_categories:
        msg = "category_names must have length n_categories"
        raise ValueError(msg)

    theta: dict[str, dict[str, float]] = {}
    for k, cat in enumerate(category_names):
        t: NDArray[np.float64] = rng.normal(0, strength_spread, size=n_models)
        t = np.roll(t, k * (n_models // n_categories))
        t = t - t.mean()
        theta[cat] = {m: float(t[i]) for i, m in enumerate(models)}

    raw_weights: NDArray[np.float64] = rng.dirichlet(np.ones(n_categories) * 2)
    category_weights = {cat: float(w) for cat, w in zip(category_names, raw_weights, strict=True)}

    comparisons: list[tuple[str, str, bool]] = []
    categories: list[str] = []
    for _ in range(n_comparisons):
        cat = rng.choice(category_names, p=raw_weights)
        i, j = rng.choice(n_models, 2, replace=False)
        t_cat = theta[cat]
        p = expit(t_cat[models[i]] - t_cat[models[j]])
        win = rng.random() < p
        comparisons.append((models[i], models[j], bool(win)))
        categories.append(str(cat))

    return {
        "comparisons": comparisons,
        "models": models,
        "true_strengths": theta,
        "categories": categories,
        "category_weights": category_weights,
    }


def simulate_rock_paper_scissors(
    n_models: int = 6,
    n_comparisons: int = 5000,
    cycle_strength: float = 0.8,
    transitive_strength: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate data with *irreducible* cyclic structure.

    Models have a transitive component (some are generally better)
    PLUS a cyclic component that cannot be explained away by categories.
    This is the Hodge curl in action.

    The cyclic component is constructed by assigning each model a position
    on a circle and adding a rotational advantage: models beat those
    "clockwise-adjacent" to them but lose to those "counter-clockwise".

    Args:
        n_models: Number of models to simulate.
        n_comparisons: Number of pairwise comparisons to generate.
        cycle_strength: Strength of the cyclic component.
        transitive_strength: Strength of the transitive component.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: comparisons (list of (model_a, model_b, a_wins)),
        models (list of str), true_transitive (dict {model: s_i} gradient component),
        true_curl_magnitude (float).
    """
    rng = np.random.default_rng(seed)
    models = [f"M{i}" for i in range(n_models)]

    s: NDArray[np.float64] = rng.normal(0, transitive_strength, size=n_models)
    s = s - s.mean()

    angles = np.linspace(0, 2 * np.pi, n_models, endpoint=False)
    C = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            C[i, j] = cycle_strength * np.sin(angles[i] - angles[j])

    Y = np.subtract.outer(s, s) + C

    comparisons: list[tuple[str, str, bool]] = []
    categories: list[str] = []
    for _ in range(n_comparisons):
        i, j = rng.choice(n_models, 2, replace=False)
        p = expit(Y[i, j])
        win = rng.random() < p
        comparisons.append((models[i], models[j], bool(win)))
        categories.append("general")

    curl_var = np.sum(np.triu(C, k=1) ** 2)
    total_var = np.sum(np.triu(Y, k=1) ** 2)

    return {
        "comparisons": comparisons,
        "models": models,
        "true_transitive": {m: float(s[i]) for i, m in enumerate(models)},
        "true_curl_magnitude": float(curl_var / total_var) if total_var > 0 else 0.0,
        "categories": categories,
    }


def simulate_llm_arena(
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate a realistic LLM arena scenario.

    Six models with names evoking real LLMs, three task categories
    (reasoning, creative_writing, coding) with plausible strength profiles.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Same format as simulate_heterogeneous.
    """
    rng = np.random.default_rng(seed)
    models = ["AlphaLM", "BetaChat", "GammaCoder", "DeltaWrite", "EpsilonAll", "ZetaMath"]
    categories = ["reasoning", "creative_writing", "coding"]

    theta: dict[str, dict[str, float]] = {
        "reasoning": {
            "AlphaLM": 0.8,
            "BetaChat": -0.2,
            "GammaCoder": -0.4,
            "DeltaWrite": -1.5,
            "EpsilonAll": 0.3,
            "ZetaMath": 2.0,
        },
        "creative_writing": {
            "AlphaLM": -0.3,
            "BetaChat": 1.2,
            "GammaCoder": -1.5,
            "DeltaWrite": 2.0,
            "EpsilonAll": 0.5,
            "ZetaMath": -1.2,
        },
        "coding": {
            "AlphaLM": 0.2,
            "BetaChat": -1.0,
            "GammaCoder": 2.2,
            "DeltaWrite": -1.0,
            "EpsilonAll": 0.3,
            "ZetaMath": 0.4,
        },
    }

    category_weights = {"reasoning": 0.35, "creative_writing": 0.35, "coding": 0.30}

    n_comparisons = 8000
    comparisons: list[tuple[str, str, bool]] = []
    cats: list[str] = []
    probs = [category_weights[c] for c in categories]
    for _ in range(n_comparisons):
        cat = rng.choice(categories, p=probs)
        i, j = rng.choice(len(models), 2, replace=False)
        mi, mj = models[i], models[j]
        p = expit(theta[cat][mi] - theta[cat][mj])
        win = rng.random() < p
        comparisons.append((mi, mj, bool(win)))
        cats.append(str(cat))

    return {
        "comparisons": comparisons,
        "models": models,
        "true_strengths": theta,
        "categories": cats,
        "category_weights": category_weights,
    }
