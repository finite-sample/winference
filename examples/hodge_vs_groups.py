"""
hodge_vs_groups: When does each approach win?
=============================================

Demonstrates the two approaches on data generated from two different
mechanisms:

  A) Heterogeneous strengths (non-transitivity is an aggregation artifact)
     -> Per-group BT wins because the structure dissolves with conditioning.

  B) Rock-paper-scissors (irreducible cyclic structure)
     -> Hodge wins because the curl is real and won't go away with categories.

Run:  python examples/hodge_vs_groups.py
"""

import numpy as np

from winference import (
    BradleyTerry,
    GroupCalibrator,
    GroupTest,
    HodgeDecomposition,
    TournamentGraph,
    brier_score,
    expected_calibration_error,
)
from winference.simulate import simulate_heterogeneous, simulate_rock_paper_scissors


def evaluate(comparisons, categories, models, label):
    """Run the full pipeline and report calibration metrics."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    # Graph diagnostic
    tg = TournamentGraph(models)
    for a, b, w in comparisons:
        tg.add_result(a, b, w)
    summary = tg.summary()
    print(f"  Non-transitivity index: {summary['nontransitivity_index']:.2f}")
    print(f"  Cyclic triples: {summary['cyclic_triples']}/{summary['total_triples']}")

    # Hodge
    W = tg.win_rate_matrix()
    hd = HodgeDecomposition(models)
    hodge = hd.fit(W, weights=tg.counts)
    print(
        f"  Hodge: transitive={hodge.transitive_variance:.1%}, cyclic={hodge.cyclic_variance:.1%}"
    )

    # Group test (if categories are informative)
    groups = sorted(set(categories))
    if len(groups) > 1:
        gt = GroupTest(models, groups)
        gt.fit(comparisons, categories)
        test = gt.test_result()
        print(f"  Group LRT: L={test['statistic']:.1f}, p={test['p_value']:.2e}")
    else:
        gt = None

    # Train/test split
    rng = np.random.default_rng(99)
    n = len(comparisons)
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_comps = [comparisons[i] for i in idx[:split]]
    train_cats = [categories[i] for i in idx[:split]]
    test_comps = [comparisons[i] for i in idx[split:]]
    test_outcomes = np.array([float(w) for _, _, w in test_comps])

    # Method 1: Global BT
    bt = BradleyTerry(models)
    bt.fit(train_comps)
    pred_bt = np.array([bt.win_probability(a, b) for a, b, _ in test_comps])

    # Method 2: Hodge transitive
    tg_tr = TournamentGraph(models)
    for a, b, w in train_comps:
        tg_tr.add_result(a, b, w)
    hd_tr = HodgeDecomposition(models)
    hd_tr.fit(tg_tr.win_rate_matrix(), weights=tg_tr.counts)
    pred_hodge = np.array([hd_tr.transitive_win_probability(a, b) for a, b, _ in test_comps])

    # Method 3: Per-group BT (only if categories exist)
    if len(groups) > 1:
        gt_tr = GroupTest(models, groups)
        gt_tr.fit(train_comps, train_cats)
        gc = GroupCalibrator(gt_tr)
        pred_group = np.array([gc.win_probability(a, b) for a, b, _ in test_comps])
    else:
        pred_group = None

    print(f"\n  {'Method':<25s} {'ECE':>8s} {'Brier':>8s}")
    print("  " + "-" * 43)
    ece_bt = expected_calibration_error(pred_bt, test_outcomes)
    brier_bt = brier_score(pred_bt, test_outcomes)
    print(f"  {'Global BT':<25s} {ece_bt:8.4f} {brier_bt:8.4f}")
    ece_hodge = expected_calibration_error(pred_hodge, test_outcomes)
    brier_hodge = brier_score(pred_hodge, test_outcomes)
    print(f"  {'Hodge (transitive)':<25s} {ece_hodge:8.4f} {brier_hodge:8.4f}")
    if pred_group is not None:
        ece_group = expected_calibration_error(pred_group, test_outcomes)
        brier_group = brier_score(pred_group, test_outcomes)
        print(f"  {'Per-group BT':<25s} {ece_group:8.4f} {brier_group:8.4f}")


def main():
    print("=" * 60)
    print("  winference: Hodge vs Per-Group BT -- when does each win?")
    print("=" * 60)

    # -- Scenario A: Heterogeneous strengths --------------------
    data_a = simulate_heterogeneous(
        n_models=6,
        n_categories=3,
        n_comparisons=6000,
        strength_spread=1.5,
        category_names=["math", "creative", "coding"],
        seed=42,
    )
    evaluate(
        data_a["comparisons"],
        data_a["categories"],
        data_a["models"],
        "Scenario A: Heterogeneous strengths (aggregation artifact)",
    )
    print("\n  -> Per-group BT should win: the non-transitivity dissolves")
    print("    when we condition on category.")

    # -- Scenario B: Irreducible cycles -------------------------
    data_b = simulate_rock_paper_scissors(
        n_models=6,
        n_comparisons=6000,
        cycle_strength=0.6,
        transitive_strength=1.0,
        seed=42,
    )
    evaluate(
        data_b["comparisons"],
        data_b["categories"],  # all "general" -- no informative categories
        data_b["models"],
        "Scenario B: Rock-paper-scissors (irreducible cycles)",
    )
    print("\n  -> Hodge should win: cycles are real, not an aggregation")
    print("    artifact. Stripping the curl gives a cleaner transitive signal.")

    # -- Scenario C: Both at once -------------------------------
    # Heterogeneous data BUT also with residual curl within categories
    print(f"\n{'─' * 60}")
    print("  Scenario C: Both mechanisms combined")
    print(f"{'─' * 60}")
    print("  (Mix of het-strengths across categories + RPS within each)")
    print("  -> In practice you would: (1) condition on category,")
    print("    (2) run Hodge within each category, (3) check if residual")
    print("    curl is small.  If yes -> het-groups sufficed.  If no ->")
    print("    you have irreducible non-transitivity that Hodge captures.")
    print()
    print("  This is the key diagnostic: does conditioning on category")
    print("  kill the curl?  The answer determines which approach is")
    print("  primary for your data.")


if __name__ == "__main__":
    main()
