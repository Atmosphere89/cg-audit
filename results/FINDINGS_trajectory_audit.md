# Claim-vs-outcome audit of public agent trajectories (2026-07-07)

Zero-generation-cost audit of a **public, real-world agent trajectory corpus**:
[SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories)
— OpenHands (gpt-4o-2024-08-06) runs on SWE-Gym tasks, each with a
harness-executed `resolved` label. We compare the agent's **final-message claim**
against that label. n = 1500 trajectories, wall ≈ 20 s, cost $0.

## Headline

> **FCR_strong = 79.1%** — of the 416 trajectories whose final message makes a
> *strong* completion claim ("the issue has been resolved", "successfully
> fixed", "all tests pass"), **4 out of 5 were in fact not resolved.**

| final-message class | share | unresolved given class |
|---|---|---|
| strong claim | 27.7% | **79.1%** |
| none | 70.2% | 98.6% |
| hedged / admit | ~2% | ~100% |

Stable across all 7 run configurations (per-run FCR 0.77–0.85).

## The honest two-sided reading

1. **Claims carry real signal**: P(resolved | strong claim) = 20.9% vs
   P(resolved | no claim) = 1.4% and base rate ≈ 7.1% — a ~3× lift. The agent
   is not babbling at random.
2. **And they are grossly unreliable as a success indicator**: 79% of confident
   completion assertions are false. Any pipeline that trusts the agent's final
   message as a done-signal — UI status, auto-merge, downstream orchestration —
   inherits a 4-in-5 false-positive rate on this distribution.

This replicates, on **someone else's real production-style corpus**, the same
verbal-behavioral gap we measured in BS-Bench replication (CG = 92.5 pp) and in
our own 5-task demo — third dataset, same phenomenon, now at n=1500 for free.

## Audit-the-audit

- Rule-based claim classifier v0 with per-trajectory evidence strings retained
  (`results/trajectory_audit_rows.jsonl`).
- Manual inspection of sampled false-success rows: 5/6 were genuine strong
  claims; 1 exposed a hedge-prefix miss ("the issue *might* have been
  resolved" matched as strong). Classifier hardened (hedging modal within 24
  chars downgrades to hedged); 14 rows reclassified; **headline moved by
  +0.02 pp** — robust to the fix.

## Caveats

- One corpus, one agent scaffold (OpenHands), one base model (gpt-4o-2024-08-06),
  first 1500 of the stream (not a random sample of the full set).
- Final-message-only claim extraction (last 3 assistant texts); rule-based v0.
- `resolved` is the harness's oracle; weak-test caveats from the SWE-bench
  correctness literature apply to the label itself.
- These are *sampled* (unfiltered) trajectories used for verifier training —
  the companion SFT set filters resolved=True, so false-success claims mostly
  do not flow into SFT *there*. Corpora curated by LLM-judge instead of
  execution (e.g. some distillation sets) are the natural next audit target.

## Reproduce

`python trajectory_audit.py` (streams from HF; no API keys needed).
