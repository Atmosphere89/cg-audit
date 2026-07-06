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

## Addendum (2026-07-07): cross-corpus / cross-model replication (n≈3,400 more, $0)

Same audit on three more public corpora (different curation pipelines, different
base models), plus one SFT corpus without labels:

| corpus / agent model | n | resolved | strong-claim share | **FCR_strong** | lift P(res\|strong)/P(res\|none) |
|---|---|---|---|---|---|
| SWE-Gym / gpt-4o (reference) | 1500 | 7.1% | 27.7% | **79.1%** | **~3.0×** |
| SWE-rebench (nebius) | 1000 | 49.2% | 42.8% | **48.8%** | 1.07× |
| Open-SWE-Traces / Minimax-M2.5 | 800 | 52.8% | 40.1% | **45.8%** | 1.06× |
| Open-SWE-Traces / Qwen3.5-122B | 800 | 42.6% | 78.2% | **57.8%** | **0.96×** |
| SWE-Hero (SFT corpus, no labels) | 800 | — | 54.4% | — | — |

Findings:

1. **The false-claim phenomenon is universal, not a gpt-4o quirk.** Across four
   corpora and four agent models, 46–79% of strong completion claims are false.
2. **Worse: for the newer models the claim channel is *uninformative*.** The
   gpt-4o run's claims at least carried a 3× lift. For SWE-rebench and
   Minimax the lift is ~1.06×; for **Qwen3.5-122B it is 0.96× — a confident
   "the issue has been resolved" tells you literally nothing (slightly less
   than nothing) about whether it was.** Claim style is a model personality
   trait (Qwen asserts completion in 78% of trajectories, gpt-4o in 28%),
   essentially decoupled from competence.
3. **Weak-oracle quantified (nebius cross-tab):** when the agent's own
   *generated* tests pass, the real oracle still fails **48.7%** of the time —
   a coin flip. And seeing their own tests pass makes agents claim *more*
   (strong-claim share 37.6% → 53.1%) while still being wrong 43% of the time:
   self-generated evidence inflates confidence without proportionate accuracy —
   the implicit-reward-hacking shape, measured in the wild.
4. **SFT leakage surface:** 54.4% of SWE-Hero's training targets end in strong
   completion claims, and the released data carries no execution labels. Given
   (2), any corpus not curated by execution should be presumed to contain false
   completion claims as *training targets* — distilling claim-inflation.

Caveats: FCR is base-rate sensitive (resolved rates differ 7%→53% across
corpora); the lift column is the base-rate-robust comparison. Claim patterns
were developed on gpt-4o phrasing; per-model phrasing coverage may differ
(Qwen's 78% share suggests coverage is adequate). First-N sampling, not random.
