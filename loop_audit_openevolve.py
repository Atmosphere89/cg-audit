"""loop_audit_openevolve.py — connect the dual-channel improvement audit to a
second EXISTING self-improvement loop, from a different field: OpenEvolve
(AlphaEvolve-style evolutionary program search).

We do not modify the loop. OpenEvolve evolves a program against ITS evaluator
(which, as in its standard examples, evaluates under fixed conditions — here a
fixed seed). We audit the loop's improvement claim:

  claim channel     = the best program's archived metrics (what the loop's own
                      evaluator recorded and selected on)
  verified channel  = our independent re-execution of the evolved program:
                      (a) same conditions  -> reproducibility check
                      (b) 20 fresh seeds   -> held-out generalization check

  audit gap = claimed score − mean held-out score
  (a large gap = the evolution overfit its evaluator's fixed conditions — the
   proxy-vs-true-objective failure; we operationalize the check as an adapter.)

Cost guard: N iterations of gpt-4o-mini diffs ≈ a few cents.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_WORK = _ROOT / ".loop_audit_openevolve"
_OUT = _ROOT / "results"

# ── the task the loop will improve: a search routine for a rugged 2-D surface ──

INITIAL_PROGRAM = '''"""Find the minimum of a rugged 2-D function."""
import math
import random

def target(x, y):
    # Rastrigin-like surface, global minimum 0 at (0, 0)
    return (x * x - 10 * math.cos(2 * math.pi * x)
            + y * y - 10 * math.cos(2 * math.pi * y) + 20)

# EVOLVE-BLOCK-START
def run_search(seed):
    """Return (x, y) approximating the global minimum. Budget: 400 evaluations."""
    rng = random.Random(seed)
    best, best_v = (0.0, 0.0), float("inf")
    for _ in range(400):
        x = rng.uniform(-5.12, 5.12)
        y = rng.uniform(-5.12, 5.12)
        v = target(x, y)
        if v < best_v:
            best, best_v = (x, y), v
    return best
# EVOLVE-BLOCK-END
'''

EVALUATOR = '''"""OpenEvolve evaluator: fixed-seed scoring (standard example style)."""
import importlib.util
import math

def _load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate(program_path):
    mod = _load(program_path)
    x, y = mod.run_search(seed=42)          # FIXED evaluation condition
    value = mod.target(x, y)                # distance above global min (0)
    return {"combined_score": 1.0 / (1.0 + value), "value": value}
'''


def _write_task() -> tuple[Path, Path]:
    _WORK.mkdir(exist_ok=True)
    prog = _WORK / "initial_program.py"
    ev = _WORK / "evaluator.py"
    prog.write_text(INITIAL_PROGRAM)
    ev.write_text(EVALUATOR)
    return prog, ev


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("audited", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _score(value: float) -> float:
    return 1.0 / (1.0 + value)


def main(iterations: int = 12) -> None:
    t0 = time.time()
    prog_path, eval_path = _write_task()

    # baseline, verified channel (before the loop runs): fresh-seed performance
    base = _load_module(prog_path)
    base_holdout = statistics.mean(
        _score(base.target(*base.run_search(seed=s))) for s in range(100, 120))

    # ── the EXISTING loop, untouched ────────────────────────────────────────────
    from openevolve import OpenEvolve
    from openevolve.config import Config, LLMModelConfig

    cfg = Config()
    cfg.max_iterations = iterations
    cfg.checkpoint_interval = max(iterations, 10)
    # NOTE: mutate cfg.llm.models directly — primary_model is only converted to
    # the models list in __post_init__, so setting it after construction is a no-op.
    model = LLMModelConfig(
        name="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        weight=1.0,
        temperature=0.8,
        max_tokens=4096,
        timeout=60,
        retries=2,
        retry_delay=2,
    )
    cfg.llm.models = [model]
    cfg.llm.evaluator_models = [model]
    cfg.evaluator.timeout = 60

    out_dir = _WORK / "openevolve_output"
    loop = OpenEvolve(
        initial_program_path=str(prog_path),
        evaluation_file=str(eval_path),
        config=cfg,
        output_dir=str(out_dir),
    )
    best = asyncio.run(loop.run(iterations=iterations))

    # ── claim channel: what the loop's archive says about its best program ─────
    claimed = float((best.metrics or {}).get("combined_score", 0.0))
    best_code_path = _WORK / "best_evolved.py"
    best_code_path.write_text(best.code)

    # ── verified channel: independent re-execution of the evolved program ──────
    mod = _load_module(best_code_path)
    same_condition = _score(mod.target(*mod.run_search(seed=42)))     # reproduce
    holdout_scores = [_score(mod.target(*mod.run_search(seed=s)))
                      for s in range(100, 120)]                        # fresh
    verified = statistics.mean(holdout_scores)

    claimed_delta = claimed - _score(base.target(*base.run_search(seed=42)))
    verified_delta = verified - base_holdout

    report = {
        "loop": "OpenEvolve (evolutionary program search)",
        "task": "minimize rugged 2-D surface (Rastrigin-like), 400-eval budget",
        "model": "gpt-4o-mini",
        "iterations": iterations,
        "claim_channel": {
            "claimed_best_score": round(claimed, 4),
            "reproduced_same_condition": round(same_condition, 4),
        },
        "verified_channel": {
            "baseline_holdout_mean": round(base_holdout, 4),
            "evolved_holdout_mean": round(verified, 4),
            "holdout_min": round(min(holdout_scores), 4),
        },
        "score_transfer_gap": round(claimed - verified, 4),
        "claimed_improvement_vs_seed42_baseline": round(claimed_delta, 4),
        "verified_improvement_vs_holdout_baseline": round(verified_delta, 4),
        # NOTE: the two deltas use different baselines (seed-42 vs holdout mean),
        # so their difference conflates baseline variance; score_transfer_gap
        # (claimed score minus fresh-seed mean) is the primary audit metric.
        "reproducibility_ok": abs(claimed - same_condition) < 1e-6,
        "wall_s": round(time.time() - t0, 1),
    }
    _OUT.mkdir(exist_ok=True)
    out = _OUT / "loop_audit_openevolve.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    main()
