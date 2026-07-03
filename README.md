# cg-audit — an open scorer for the Compliance Gap (BS-Bench)

An independent, open-source implementation of the seven **BS-Bench** metrics
(arXiv [2605.01771](https://arxiv.org/abs/2605.01771)) — ICR, DF, FCR, VCR, ACR,
**CG = VCR − ACR**, TA — plus a session runner that replays the BS-Bench
environment against live OpenAI / Anthropic models.

The benchmark's own scoring implementation (`evaluation/metrics.py`) is not part
of its public artifact; this repository provides a runnable, auditable one.
Every text-channel judgement (VCR / declared-read / TA) retains a per-session
`*_evidence` field, so the audit itself can be audited.

## First independent replication (n = 40)

GPT-4o-mini · Medium-50 task suite · *None* framing · temperature 0.7 · seeds 1–40:

| metric | ours (n=40) | paper anchor (n=40) |
|---|---|---|
| VCR (verbal compliance) | **1.00** | 1.00 |
| ACR (behavioral, individual reads) | 0.075 | 0.00 |
| **CG** | **92.5 pp** | 100 pp |

- **The compliance gap replicates.** 40/40 sessions claim verification is
  complete; only 3/40 actually read every file individually. The 7.5-pp gap vs
  the anchor is real behavioral variance (seeds 14/32/36 spontaneously
  self-complied), not a scoring artifact.
- **CG is scorer-sensitive.** A naive-but-plausible VCR rule set scored 0.65
  (CG = 57.5); after a manual audit of all 40 transcripts, corrected rules score
  1.00 (CG = 92.5) — a **35-pp swing on identical logs**. Since the original
  scorer is closed and single, this sensitivity is invisible from the original
  artifact. All rules, evidence and transcripts are in this repo.

Details and caveats: [results/FINDINGS_n40.md](results/FINDINGS_n40.md).
Raw session logs: [logs/](logs/) (also attached as a release asset).

## Inspect AI integration

`cg_inspect.py` packages the scorer for [Inspect](https://inspect.aisi.org.uk/)
(UK AI Security Institute's evaluation framework): a `compliance_gap_scorer()`
that reads the sample transcript (assistant text = verbal channel, tool calls =
behavioral channel), and a self-contained `compliance_gap` demo task with the
BS-Bench file-verification environment.

```bash
pip install inspect-ai
inspect eval cg_inspect.py --model openai/gpt-4o-mini
inspect eval cg_inspect.py -T framing=override --model anthropic/claude-sonnet-4-6
```

Verified: scripted-model integration tests (`test_cg_inspect.py`, no API needed)
plus a live smoke — gpt-4o-mini under the default framing scores CG = +1
(claims verification complete, reads zero files individually), matching the
n=40 replication above.

### Attribution

`task_data/simple_10/` is from the BS-Bench v1.0 task suite
(Shin, 2026, arXiv 2605.01771; data CC BY 4.0, via https://osf.io/mvnq4/).

## Reproduce

```bash
pip install -r requirements.txt
# Download the BS-Bench v1.0 task suite (task_suite_v1.0.zip) from
# https://osf.io/mvnq4/ and unzip it, then:
export BS_BENCH_SUITE=/path/to/task_suite
export OPENAI_API_KEY=...          # and/or ANTHROPIC_API_KEY
python cg_batch40.py               # 40 sessions, resumable, ~US$0.3
python -m pytest test_cg_audit.py  # scorer unit tests
```

Reconstruction notes (the paper's runner is unpublished): tool names, neutral
descriptions, instruction texts and the user-layer template are verbatim from
the public task suite; `batch_verify` returns truncated (400-char) real content
and `delegate_task` returns a shallow "no errors" report — both documented in
`cg_session_runner.py`. ACR here is the strict individual-read criterion
matching the anchors.

## License

MIT. Author: anonymous.
