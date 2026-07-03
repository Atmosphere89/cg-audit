# Connecting the dual-channel audit to existing self-improvement loops (2026-07-04)

Goal: connect the claim-vs-verification audit to two EXISTING self-improvement
loops from different fields, **without modifying the loops**. The adapter
pattern is identical in both:

- **claim channel** — the metric the loop itself reports/selects on
  (optimizer-visible validation score; evolution-archive fitness)
- **verified channel** — our independent re-execution of the loop's product
  under conditions the loop never saw (held-out split; fresh seeds)
- **audit metric** — does the claimed gain transfer? (`score_transfer_gap`,
  `audit_gap_pp`)

Adapters: `loop_audit_dspy.py`, `loop_audit_openevolve.py`. Total spend for all
runs: well under US$0.10 (gpt-4o-mini / gpt-3.5-turbo).

## Connection 1 — OpenEvolve (evolutionary program search; algorithm-discovery field)

Task: evolve a 400-evaluation search routine minimizing a Rastrigin-like 2-D
surface. The loop's evaluator scores under a **fixed seed** (standard example
style) — a realistic overfitting hazard.

| channel | result |
|---|---|
| claim (archive best) | combined_score **1.0000** (exact global minimum) |
| reproduce, same condition | 1.0000 — reproducibility OK |
| verified (20 fresh seeds) | mean **1.0000**, worst seed 1.0000 |
| **score_transfer_gap** | **0.0** |

Verdict: the loop's improvement (random search → reliably exact optimum in 12
generations) is genuine and **fully transfers** to unseen conditions — an
honest loop, correctly given a clean bill by the audit.

Two incidental findings:
1. **Audit robustness**: in an earlier run where the loop's LLM calls all
   failed (config bug — `primary_model` set after construction bypasses
   `__post_init__`), the adapter still returned a correct "no improvement /
   reproducible / gap 0" report. The audit degrades gracefully when the loop
   breaks.
2. **Metric-definition bug found & fixed in the adapter itself**: comparing
   deltas against *different baselines* (fixed-seed vs holdout-mean) produced a
   spurious 0.17 "gap" on a perfect program. Primary metric is now the direct
   `score_transfer_gap` (claimed score − fresh-seed mean). Even the auditor
   needs auditing; the JSON keeps both values.

## Connection 2 — DSPy GEPA (reflective prompt evolution; prompt-optimization field)

Task: 5-way intent classification (synthetic, deliberately confusable labels),
20 train / 20 val (loop-visible) / 20 holdout (loop-invisible).

| attempt | design | outcome |
|---|---|---|
| 1 | unconstrained `text -> label` | baseline 0% — model invents its own label taxonomy; GEPA cannot climb from all-zero scores |
| 2 | constrained 3-class signature | baseline 100% — nothing to improve ("all subsample scores perfect, skipping") |
| 3 | 5 confusable classes | val 1.0 / holdout 0.95 — still no improvement window (gpt-4o-mini) |
| 4 | weaker task LM (gpt-3.5-turbo) | val 1.0 / holdout 0.95 — same |

Final report: claimed improvement 0.0, verified improvement 0.0,
**audit_gap 0.0 pp** — the loop claimed nothing and delivered nothing beyond an
already-perfect baseline; claim and reality agree.

Honest reading: the **connection works end-to-end** (both channels measured,
report emitted), but this task family gave GEPA no improvement window, so the
audit's interesting case (a non-zero claimed gain, checked for transfer) was
not exercised here. The four attempts are themselves the finding: *the audit is
only as informative as the loop's improvement window, and constructing a task
inside that window is the hard part of any self-improvement demo.* A meaningful
DSPy audit needs a naturally hard task (e.g., a real dataset where baselines sit
at 60–80%), which costs more than this demo budget.

## Shared conclusion

One adapter pattern connected, unmodified, to two loops from unrelated fields
(prompt optimization / algorithm evolution). The pattern generalizes because
every self-improvement loop has the same two channels: what it says it achieved,
and what its product actually does when re-executed on conditions it never saw.
