winference
==========

**Win rate calibration under non-transitivity.**

When you run an LLM arena and report "Model A beats Model B 62% of the time,"
is that number *calibrated*?  And does it still hold when your users ask
different questions than your evaluation set?

If model strengths vary across task types, aggregate win rates can exhibit
**non-transitive** preferences: A beats B, B beats C, but C beats A.  Standard
Bradley-Terry / Elo assumes this doesn't happen, and when it does, your
calibration breaks — especially under distribution shift.

``winference`` provides two approaches to calibrating win rates in the presence of
non-transitivity, plus diagnostics to decide which one you need.

Installation
------------

.. code-block:: bash

   pip install winference

The Two Approaches
------------------

A) Hodge Decomposition
~~~~~~~~~~~~~~~~~~~~~~

Decomposes the pairwise comparison matrix into:

- **Gradient** (transitive): a potential ``s_i`` per model such that the
  log-odds ≈ ``s_i − s_j``.  This part *can* be calibrated to a scalar ranking.
- **Curl** (cyclic): rock-paper-scissors structure that *cannot* be represented
  by any linear ranking.

Calibrate win rates from the gradient component.  Report the curl fraction as
the share of variance your calibration ignores.

**Use when:** Cycles persist even after conditioning on task category — the
non-transitivity is irreducible.

B) Heterogeneous Groups
~~~~~~~~~~~~~~~~~~~~~~~

Test whether model strengths differ across prompt categories (math, creative,
coding, ...) using a **likelihood-ratio test**.  If so, fit Bradley-Terry per
category.  Win rates for any target distribution are then:

.. math::

   P(A > B | \pi^*) = \sum_k  \pi^*_k \cdot \sigma(\theta_{A,k} - \theta_{B,k})

This gives you **composable** calibration: swap in any target distribution
without refitting.

**Use when:** Non-transitivity dissolves when you condition on prompt category.

Quickstart
----------

.. code-block:: python

   from winference import (
       TournamentGraph, BradleyTerry, HodgeDecomposition,
       GroupTest, GroupCalibrator, expected_calibration_error,
   )
   from winference.simulate import simulate_llm_arena

   # 1. Simulate (or load) arena data
   data = simulate_llm_arena()
   comparisons = data["comparisons"]
   categories  = data["categories"]
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

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/tournament
   api/bradley_terry
   api/hodge
   api/groups
   api/calibration
   api/simulate


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
