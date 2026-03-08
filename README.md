# winference

**Win rate calibration under non-transitivity.**

When you run an LLM arena and report "Model A beats Model B 62% of the time,"
is that number *calibrated*?  And does it still hold when your users ask
different questions than your evaluation set?

If model strengths vary across task types, aggregate win rates can exhibit
**non-transitive** preferences: A beats B, B beats C, but C beats A.  Standard
Bradley-Terry / Elo assumes this doesn't happen, and when it does, your
calibration breaks — especially under distribution shift.

`winference` provides two approaches to calibrating win rates in the presence of
non-transitivity, plus diagnostics to decide which one you need.

---

## The two approaches

### A) Hodge decomposition → calibrate the transitive signal

Decomposes the pairwise comparison matrix into:

- **Gradient** (transitive): a potential `s_i` per model such that the
  log-odds ≈ `s_i − s_j`.  This part *can* be calibrated to a scalar ranking.
- **Curl** (cyclic): rock-paper-scissors structure that *cannot* be represented
  by any linear ranking.

Calibrate win rates from the gradient component.  Report the curl fraction as
the share of variance your calibration ignores.

**Use when:** Cycles persist even after conditioning on task category — the
non-transitivity is irreducible.

### B) Heterogeneous groups → calibrate per category, compose

Test whether model strengths differ across prompt categories (math, creative,
coding, ...) using a **likelihood-ratio test**.  If so, fit Bradley-Terry per
category.  Win rates for any target distribution are then:

```
P(A > B | π*) = Σ_k  π*_k · σ(θ_{A,k} − θ_{B,k})
```

This gives you **composable** calibration: swap in any target distribution
without refitting.

**Use when:** Non-transitivity dissolves when you condition on prompt category.

---

## Quickstart

```bash
pip install numpy scipy scikit-learn matplotlib
# Then from the repo root:
pip install -e .
```

```python
from winference import (
    TournamentGraph, BradleyTerry, HodgeDecomposition,
    GroupTest, GroupCalibrator, expected_calibration_error,
)
from winference.simulate import simulate_llm_arena

# 1. Simulate (or load) arena data
data = simulate_llm_arena()
comparisons = data["comparisons"]   # list of (model_a, model_b, a_wins)
categories  = data["categories"]    # list of category labels per comparison
models      = data["models"]

# 2. Graph triage: is non-transitivity a problem?
tg = TournamentGraph(models)
for a, b, w in comparisons:
    tg.add_result(a, b, w)

print(tg.summary())
# → {'nontransitivity_index': 0.83, 'cyclic_triples': 7, ...}

# 3a. Hodge decomposition
hd = HodgeDecomposition(models)
result = hd.fit(tg.win_rate_matrix(), weights=tg.counts)
print(f"Transitive: {result.transitive_variance:.0%}")
print(f"Cyclic:     {result.cyclic_variance:.0%}")

# Calibrated win probability (transitive component only)
p = hd.transitive_win_probability("ZetaMath", "DeltaWrite")

# 3b. Group heterogeneity test
groups = sorted(set(categories))
gt = GroupTest(models, groups)
gt.fit(comparisons, categories)
print(gt.test_result())
# → {'statistic': 342.1, 'p_value': 1.2e-63, 'reject_at_05': True}

# Composable win rates
gc = GroupCalibrator(gt)
p_math_heavy = gc.win_probability(
    "ZetaMath", "DeltaWrite",
    target_distribution={"reasoning": 0.7, "creative_writing": 0.15, "coding": 0.15},
)
p_creative_heavy = gc.win_probability(
    "ZetaMath", "DeltaWrite",
    target_distribution={"reasoning": 0.15, "creative_writing": 0.7, "coding": 0.15},
)
```

See `examples/quickstart.py` for the full pipeline with calibration comparison
and reliability diagrams.

---

## The diagnostic pipeline

```
┌─────────────────────────────┐
│  Build tournament graph     │
│  Run Tarjan's SCC           │
└──────────┬──────────────────┘
           │
     All SCCs size 1?
      ╱           ╲
    YES            NO
     │              │
  Standard BT    ┌──┴───────────────────────┐
  is fine        │  Condition on categories  │
                 │  Check: do SCCs shrink?   │
                 └──────┬───────────────┬────┘
                    YES                  NO
                     │                    │
              ┌──────┴──────┐    ┌───────┴────────┐
              │ Per-group   │    │ Hodge decomp   │
              │ BT + LRT    │    │ calibrate grad │
              │ + compose   │    │ report curl    │
              └─────────────┘    └────────────────┘
```

---

## Other sources of non-transitivity

Non-transitivity in pairwise comparisons doesn't always come from heterogeneous
model strengths.  Before reaching for Hodge or group decomposition, rule out
simpler causes:

| Source | What it is | How to address |
|---|---|---|
| **Judge noise** | LLM judge gives inconsistent verdicts on the same pair | Bayesian calibration (Dawid-Skene, BWRS) |
| **Position bias** | Judge prefers whichever response appears first/second | Randomise presentation order, average both orderings |
| **Style/length bias** | Judge rewards verbosity rather than quality | Regress out length/style features (cf. AlpacaEval 2.0) |
| **Evaluator disagreement** | Individual annotators are transitive, but they disagree with each other (Condorcet cycles) | Stratify by evaluator type, or accept the Hodge curl as measuring genuine disagreement |
| **Fine-grained interaction** | Strengths differ at sub-category level (algebra vs geometry within "math") | Per-prompt routing rather than aggregate ranking |
| **Context effects** | Evaluation of A vs B depends on what other models were seen | Session-aware experimental design |

`winference` targets the modelling layer (rows 4–5 in the table above). It
assumes you've already addressed measurement issues at the design/preprocessing
level.

---

## API reference

### `TournamentGraph`
Build a directed tournament graph and compute SCC structure.
- `.add_result(a, b, win)` — record one comparison
- `.strongly_connected_components()` — Tarjan's algorithm
- `.nontransitivity_index()` — fraction of models in non-trivial SCCs
- `.count_cyclic_triples()` — count A>B>C>A cycles
- `.summary()` — quick diagnostic dict

### `BradleyTerry`
Standard BT model via maximum likelihood.
- `.fit(comparisons)` — fit from (a, b, a_wins) triples
- `.win_probability(a, b)` — predicted P(a beats b)
- `.strengths()` — {model: θ}
- `.rank()` — models sorted by strength

### `HodgeDecomposition`
Hodge decomposition of the pairwise log-odds matrix.
- `.fit(W, weights)` — decompose win-rate matrix
- `.transitive_win_probability(a, b)` — P(a>b) from gradient only
- `.worst_pairs(k)` — pairs with largest cyclic residual
- `.summary()` — variance fractions (transitive / cyclic / harmonic)

### `GroupTest`
Likelihood-ratio test for heterogeneity across prompt groups.
- `.fit(comparisons, group_labels)` — fit null + per-group BT
- `.test_result()` — {statistic, df, p_value, reject_at_05}
- `.per_group_strengths()` — {group: {model: θ}}

### `GroupCalibrator`
Composable win rates from per-group BT.
- `.win_probability(a, b, target_distribution)` — composite P(a>b)
- `.sensitivity_analysis(a, b)` — how much does P(a>b) vary with π*?

### Calibration utilities
- `expected_calibration_error(predicted, observed)` — ECE
- `brier_score(predicted, observed)` — Brier score
- `reliability_diagram(predicted, observed, ax=...)` — reliability plot

### Simulators
- `simulate_transitive(...)` — pure BT (no cycles)
- `simulate_heterogeneous(...)` — per-category strengths
- `simulate_rock_paper_scissors(...)` — irreducible cyclic structure
- `simulate_llm_arena(...)` — realistic six-model arena

---

## References

- Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2011). Statistical ranking and combinatorial Hodge theory. *Mathematical Programming*, 127(1), 203–244.
- Dittrich, R., Hatzinger, R., & Katzenbeisser, W. (1998). Modelling the effect of subject-specific covariates in paired comparison studies. *Applied Statistics*, 47(4), 511–525.
- Xu, Y., Ruis, L., Rocktäschel, T., & Kirk, R. (2025). Investigating non-transitivity in LLM-as-a-Judge. *ICML 2025*.
- Li, X. & Li, S. (2025). Efficient inference for covariate-adjusted Bradley-Terry model with covariate shift. *arXiv:2503.18256*.
- Balduzzi, D., Tuyls, K., Perolat, J., & Graepel, T. (2018). Re-evaluating evaluation. *NeurIPS*.

---

## License

MIT
