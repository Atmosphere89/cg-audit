"""trajectory_audit_multi.py — cross-corpus / cross-model replication of the
claim-vs-outcome audit, plus a weak-oracle cross-tab. Zero generation cost.

Corpora:
  A nebius/SWE-rebench-openhands-trajectories  (resolved + pred_passes_gen_tests)
  B nvidia/Open-SWE-Traces  minimax_m25 split  (resolved; Minimax-M2.5 agent)
  C nvidia/Open-SWE-Traces  qwen35_122b split  (resolved; Qwen3.5-122B agent)
  D nvidia/SWE-Hero-openhands-trajectories     (NO labels -> claim prevalence
                                                in an SFT corpus only)

Questions:
  1. Does FCR_strong ≈ 0.79 (SWE-Gym/gpt-4o) replicate on other corpora and
     other base models?
  2. nebius only: when the agent's own GENERATED tests pass but the real
     oracle fails (weak-oracle cases), do strong claims get MORE confident?
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from trajectory_audit import classify_claim, final_assistant_text

_OUT = Path(__file__).resolve().parent / "results"


def _messages(rec) -> list:
    tr = rec.get("trajectory")
    if isinstance(tr, str):
        try:
            return json.loads(tr)
        except Exception:
            return []
    return tr or []


def audit_corpus(name: str, config: str | None, split: str, n_max: int,
                 resolved_key: str | None = "resolved",
                 extra_keys: tuple = ()) -> list:
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True) if config else \
        load_dataset(name, split=split, streaming=True)
    rows = []
    for i, rec in enumerate(ds):
        if i >= n_max:
            break
        text = final_assistant_text(_messages(rec))
        cls, ev = classify_claim(text)
        row = {"claim": cls, "evidence": ev}
        if resolved_key is not None:
            row["resolved"] = bool(rec.get(resolved_key))
        for k in extra_keys:
            row[k] = rec.get(k)
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}", flush=True)
    return rows


def fcr_summary(rows) -> dict:
    n = len(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["claim"]].append(r)
    strong = by.get("strong_claim", [])
    out = {"n": n,
           "strong_share": round(len(strong) / n, 3) if n else None}
    if rows and "resolved" in rows[0]:
        out["resolved_rate"] = round(sum(r["resolved"] for r in rows) / n, 4)
        if strong:
            out["FCR_strong"] = round(
                sum(1 for r in strong if not r["resolved"]) / len(strong), 4)
            out["P_resolved_given_strong"] = round(
                sum(r["resolved"] for r in strong) / len(strong), 4)
        none = by.get("none", [])
        if none:
            out["P_resolved_given_none"] = round(
                sum(r["resolved"] for r in none) / len(none), 4)
    return out


def main() -> None:
    t0 = time.time()
    report = {}

    print("=== A nebius/SWE-rebench (resolved + gen-tests) ===", flush=True)
    a = audit_corpus("nebius/SWE-rebench-openhands-trajectories", None, "train",
                     1000, extra_keys=("pred_passes_gen_tests",))
    report["nebius_swe_rebench"] = fcr_summary(a)
    # weak-oracle cross-tab
    gt_pass = [r for r in a if (r.get("pred_passes_gen_tests") or 0) >= 1.0]
    gt_fail = [r for r in a if (r.get("pred_passes_gen_tests") or 0) < 1.0]
    report["nebius_weak_oracle"] = {
        "gen_tests_pass": fcr_summary(gt_pass),
        "gen_tests_fail_or_partial": fcr_summary(gt_fail),
        "P_unresolved_given_gen_tests_pass": round(
            sum(1 for r in gt_pass if not r["resolved"]) / len(gt_pass), 4)
        if gt_pass else None,
    }
    print(json.dumps(report["nebius_swe_rebench"], indent=1), flush=True)

    print("=== B Open-SWE-Traces / Minimax-M2.5 ===", flush=True)
    b = audit_corpus("nvidia/Open-SWE-Traces", "openhands", "minimax_m25", 800)
    report["open_swe_minimax"] = fcr_summary(b)
    print(json.dumps(report["open_swe_minimax"], indent=1), flush=True)

    print("=== C Open-SWE-Traces / Qwen3.5-122B ===", flush=True)
    c = audit_corpus("nvidia/Open-SWE-Traces", "openhands", "qwen35_122b", 800)
    report["open_swe_qwen"] = fcr_summary(c)
    print(json.dumps(report["open_swe_qwen"], indent=1), flush=True)

    print("=== D SWE-Hero (SFT corpus, no labels -> prevalence only) ===", flush=True)
    d = audit_corpus("nvidia/SWE-Hero-openhands-trajectories", None, "train",
                     800, resolved_key=None)
    report["swe_hero_sft"] = fcr_summary(d)
    print(json.dumps(report["swe_hero_sft"], indent=1), flush=True)

    report["reference_swe_gym_gpt4o"] = {"FCR_strong": 0.7909, "n": 1500,
                                         "strong_share": 0.277}
    report["wall_s"] = round(time.time() - t0, 1)
    _OUT.mkdir(exist_ok=True)
    (_OUT / "trajectory_audit_multi.json").write_text(json.dumps(report, indent=2))
    print("\n=== FULL REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
