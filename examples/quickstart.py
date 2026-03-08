"""
winference quickstart
=====================

Full pipeline: Graph triage -> Hodge decomposition -> Group testing ->
Calibration comparison.

Key demonstrations:
  - Hodge:     identifies *which pairs* are badly calibrated under BT
  - Per-group: maintains calibration under distribution shift

Run:  python examples/quickstart.py
"""

import matplotlib
import numpy as np
from scipy.special import expit

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt

from winference import (
    BradleyTerry,
    GroupCalibrator,
    GroupTest,
    HodgeDecomposition,
    TournamentGraph,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)
from winference.simulate import simulate_llm_arena


def main():
    print("=" * 65)
    print("  winference quickstart: LLM arena with non-transitive win rates")
    print("=" * 65)

    # -- 1. Simulate a realistic arena -------------------------------
    data = simulate_llm_arena(seed=42)
    comparisons = data["comparisons"]
    categories = data["categories"]
    models = data["models"]

    print(f"\nModels: {models}")
    print(f"Comparisons: {len(comparisons)}")
    print(f"Categories: {sorted(set(categories))}")
    print(f"True category weights: {data['category_weights']}")

    # -- 2. Graph triage: is non-transitivity a problem? -------------
    print("\n-- Stage 1: Graph Diagnostic --")
    tg = TournamentGraph(models)
    for a, b, w in comparisons:
        tg.add_result(a, b, w)

    summary = tg.summary()
    print(f"  SCC sizes:              {tg.scc_sizes()}")
    print(f"  Non-transitivity index: {summary['nontransitivity_index']:.2f}")
    print(f"  Cyclic triples:         {summary['cyclic_triples']}/{summary['total_triples']}")

    if summary["nontransitivity_index"] == 0:
        print("  -> No non-transitivity in majority-vote graph.")
        print("    (Margins may still be heterogeneous -- check the LRT.)")
    else:
        print("  -> Non-trivial SCCs found.")
    print()

    # -- 3. Hodge decomposition --------------------------------------
    print("-- Stage 2a: Hodge Decomposition --")
    W = tg.win_rate_matrix()
    hd = HodgeDecomposition(models)
    hodge = hd.fit(W, weights=tg.counts)

    print(f"  Transitive variance:  {hodge.transitive_variance:.1%}")
    print(f"  Cyclic variance:      {hodge.cyclic_variance:.1%}")

    # -- THE HODGE VALUE PROPOSITION --
    # For high-curl pairs, BT win probabilities are systematically off.
    print("\n  * Hodge value: pair-level calibration diagnostic")

    bt_global = BradleyTerry(models)
    bt_global.fit(comparisons)

    pair_data = defaultdict(lambda: {"wins": 0, "total": 0})
    for a, b, w in comparisons:
        key = (min(a, b), max(a, b))
        if a == key[0]:
            pair_data[key]["wins"] += int(w)
        else:
            pair_data[key]["wins"] += int(not w)
        pair_data[key]["total"] += 1

    curl_errors = []
    for (a, b), d in sorted(pair_data.items()):
        obs = d["wins"] / d["total"]
        pred = bt_global.win_probability(a, b)
        i_a, i_b = hd._idx[a], hd._idx[b]
        curl_mag = abs(hodge.curl_flow[i_a, i_b])
        curl_errors.append((a, b, obs, pred, abs(obs - pred), curl_mag))

    curl_errors.sort(key=lambda x: x[5], reverse=True)

    print(f"\n  {'Pair':<30s} {'Obs':>6s} {'BT':>6s} {'|err|':>6s} {'|curl|':>6s}")
    print("  " + "-" * 56)
    print("  High-curl pairs:")
    for a, b, obs, pred, err, curl in curl_errors[:5]:
        print(f"    {a + ' vs ' + b:<28s} {obs:6.3f} {pred:6.3f} {err:6.3f} {curl:6.3f}")
    print("  Low-curl pairs:")
    for a, b, obs, pred, err, curl in curl_errors[-3:]:
        print(f"    {a + ' vs ' + b:<28s} {obs:6.3f} {pred:6.3f} {err:6.3f} {curl:6.3f}")

    curls = np.array([x[5] for x in curl_errors])
    errors = np.array([x[4] for x in curl_errors])
    corr = np.corrcoef(curls, errors)[0, 1]
    print(f"\n  Corr(|curl|, |BT error|) = {corr:.3f}")
    print("  -> Hodge tells you WHERE calibration is unreliable.\n")

    # -- 4. Group heterogeneity test ---------------------------------
    print("-- Stage 2b: Group Heterogeneity Test --")
    groups = sorted(set(categories))
    gt = GroupTest(models, groups)
    gt.fit(comparisons, categories)
    test = gt.test_result()

    print(f"  LRT statistic: {test['statistic']:.1f}")
    print(f"  Degrees of freedom: {test['df']}")
    print(f"  p-value: {test['p_value']:.2e}")
    print(f"  Reject H0 (homogeneity) at a=0.05: {test['reject_at_05']}")

    print("\n  Per-group rankings:")
    for g, strengths in gt.per_group_strengths().items():
        ranked = sorted(strengths, key=lambda m: strengths[m], reverse=True)
        print(f"    {g:20s}: {' > '.join(ranked[:3])} > ...")

    # -- THE PER-GROUP VALUE PROPOSITION --
    # Train on one distribution, test on a DIFFERENT distribution.
    print("\n-- Stage 3: Calibration Under Distribution Shift --")
    print("  * Per-group value: calibration that transfers.\n")

    rng = np.random.default_rng(123)
    n = len(comparisons)
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_comps = [comparisons[i] for i in idx[:split]]
    train_cats = [categories[i] for i in idx[:split]]

    bt_train = BradleyTerry(models)
    bt_train.fit(train_comps)

    gt_train = GroupTest(models, groups)
    gt_train.fit(train_comps, train_cats)
    gc = GroupCalibrator(gt_train)

    shift_scenarios = {
        "Same (35/35/30)": {"reasoning": 0.35, "creative_writing": 0.35, "coding": 0.30},
        "Reasoning (70/15/15)": {"reasoning": 0.70, "creative_writing": 0.15, "coding": 0.15},
        "Creative (15/70/15)": {"reasoning": 0.15, "creative_writing": 0.70, "coding": 0.15},
        "Coding (15/15/70)": {"reasoning": 0.15, "creative_writing": 0.15, "coding": 0.70},
    }

    true_theta = data["true_strengths"]

    print(f"  {'Scenario':<25s} {'BT ECE':>8s} {'Grp ECE':>8s}  {'BT Brier':>9s} {'Grp Brier':>9s}")
    print("  " + "-" * 63)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes_flat = axes.flatten()

    for si, (sname, target_dist) in enumerate(shift_scenarios.items()):
        test_comps_shifted = []
        cat_names = list(target_dist.keys())
        cat_probs = [target_dist[c] for c in cat_names]
        for _ in range(2000):
            cat = rng.choice(cat_names, p=cat_probs)
            i, j = rng.choice(len(models), 2, replace=False)
            mi, mj = models[i], models[j]
            p = expit(true_theta[cat][mi] - true_theta[cat][mj])
            win = rng.random() < p
            test_comps_shifted.append((mi, mj, bool(win)))

        test_out = np.array([float(w) for _, _, w in test_comps_shifted])
        pred_bt = np.array([bt_train.win_probability(a, b) for a, b, _ in test_comps_shifted])
        pred_grp = np.array(
            [gc.win_probability(a, b, target_dist) for a, b, _ in test_comps_shifted]
        )

        ece_bt = expected_calibration_error(pred_bt, test_out)
        ece_grp = expected_calibration_error(pred_grp, test_out)
        brier_bt = brier_score(pred_bt, test_out)
        brier_grp = brier_score(pred_grp, test_out)

        marker = " <-" if abs(ece_bt - ece_grp) > 0.005 and ece_grp < ece_bt else ""
        print(
            f"  {sname:<25s} {ece_bt:8.4f} {ece_grp:8.4f}  {brier_bt:9.4f} {brier_grp:9.4f}{marker}"
        )

        ax = axes_flat[si]
        reliability_diagram(pred_bt, test_out, n_bins=10, ax=ax, label="Global BT", color="#d95f02")
        reliability_diagram(
            pred_grp, test_out, n_bins=10, ax=ax, label="Per-group BT", color="#1b9e77"
        )
        ax.set_title(sname, fontsize=10)

    fig.suptitle(
        "Calibration Under Distribution Shift\n(trained on balanced 35/35/30, tested on shifted)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig("quickstart_reliability.png", dpi=150, bbox_inches="tight")
    print("\n  Saved -> quickstart_reliability.png")
    print("  Note: under shift, Global BT predictions compress into a narrow")
    print("  range (orange spans less of the x-axis) because it doesn't know")
    print("  the distribution changed. Per-group adapts and spreads predictions")
    print("  across the full range -- covering more of the x-axis.")

    # -- 5. The punchline -------------------------------------------
    print("\n-- Punchline --")
    p_global = bt_train.win_probability("ZetaMath", "DeltaWrite")
    print(f"  Global BT says ZetaMath vs DeltaWrite: {p_global:.3f} (always)")
    for name, dist in shift_scenarios.items():
        p = gc.win_probability("ZetaMath", "DeltaWrite", dist)
        print(f"  Per-group under {name:<25s}: {p:.3f}")
    print()
    print("  -> Global BT: ONE number regardless of who's asking.")
    print("  -> Per-group:  the RIGHT number for each use case.")
    print("  -> Hodge:      which pairs to DISTRUST in any ranking.")
    print("\nDone.")


if __name__ == "__main__":
    main()
