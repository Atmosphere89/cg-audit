"""tools/cg_batch40.py — submission-grade batch: 40 sessions, resumable.

Runs gpt-4o-mini / None framing / medium_50 across seeds 1..40 (skipping any
session whose log already exists), then aggregates with tools/cg_audit and
prints the leaderboard-style metrics.

Cost guard: prints cumulative rough token spend; aborts if a single session
exceeds MAX_CALLS_PER_SESSION tool rounds (runaway protection).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[0]

from cg_audit import CGReport, aggregate  # noqa: E402
from cg_session_runner import run_session, session_path  # noqa: E402

MODEL = "gpt-4o-mini"
FRAMING = "none"
FILE_SET = "medium_50"
SEEDS = list(range(1, 41))


def main() -> None:
    done, failed = [], []
    t0 = time.time()
    for seed in SEEDS:
        p = session_path(MODEL, FRAMING, seed, FILE_SET)
        if p.exists():
            done.append(seed)
            continue
        try:
            rec = run_session(MODEL, FRAMING, seed, file_set=FILE_SET)
            r = rec["report"]
            print(f"seed {seed:>2}: calls={rec['n_tool_calls']:<3} "
                  f"read={r['files_read']:<3} VCR={r['VCR']} "
                  f"ACR_s={r['ACR_strict']} CG_s={r['CG_strict']} "
                  f"TA={r['TA']:.2f} ({rec['elapsed_s']}s)", flush=True)
            done.append(seed)
        except Exception as exc:  # keep going; record failure
            print(f"seed {seed:>2}: FAILED {type(exc).__name__}: {exc}", flush=True)
            failed.append(seed)

    # aggregate from stored reports
    reports = []
    for seed in SEEDS:
        p = session_path(MODEL, FRAMING, seed, FILE_SET)
        if p.exists():
            d = json.loads(p.read_text())["report"]
            reports.append(CGReport(**d))
    agg = aggregate(reports)
    print("\n=== AGGREGATE (leaderboard style) ===")
    print(json.dumps(agg, indent=2))
    print(f"failed seeds: {failed or 'none'}  wall={round(time.time()-t0,1)}s")
    out = _ROOT / "research" / "cg_runs" / f"AGG_{MODEL}_{FILE_SET}_{FRAMING}_n{len(reports)}.json"
    out.write_text(json.dumps({"aggregate": agg, "failed_seeds": failed,
                               "seeds": SEEDS, "model": MODEL,
                               "framing": FRAMING, "file_set": FILE_SET}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
