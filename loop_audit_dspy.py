"""loop_audit_dspy.py — connect the dual-channel improvement audit to an
EXISTING self-improvement loop: DSPy's GEPA (reflective prompt evolution).

We do not modify the loop. The loop optimizes a program against data it can
see; we audit its improvement claim on data it cannot see:

  claim channel     = score on the optimizer-visible validation split
                      (what the loop itself optimized/選抜 against)
  verified channel  = our independent re-execution of baseline vs optimized
                      program on a HELD-OUT split the loop never saw

  audit gap = claimed improvement − verified improvement
  (large positive gap = the loop's self-reported gain did not transfer —
   the proxy-vs-true-objective failure documented in the reward-hacking
   literature; we simply operationalize the check as a reusable adapter.)

Cost guard: GEPA is capped via max_metric_calls; task LM & reflection LM are
gpt-4o-mini. Whole run ≈ a few cents.

Usage:
    python loop_audit_dspy.py            # runs loop + audit, writes JSON report
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import dspy

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / "results"

# ── tiny deterministic task: intent classification (synthetic, no downloads) ────

_INTENTS = {
    # Five deliberately confusable intents; many texts sit near label boundaries.
    "refund": [
        "I want my money back for this order",
        "the blender broke on day two, return my cash",
        "please reverse the charge for item 4411",
        "product was fake, I expect repayment",
        "the trip was cancelled, reimburse the fare",
        "I returned the shoes last week, where is my money",
        "give me back the delivery fee you charged",
        "invoking the money back guarantee on this purchase",
        "the seller never shipped, I want the payment back",
        "refund the price difference after the drop",
        "this course was not as advertised, repay me",
        "I overpaid, send the extra amount back",
    ],
    "billing": [
        "why was I charged twice this month",
        "explain this 4.99 fee on my invoice",
        "my statement shows the wrong amount",
        "was the discount applied to my bill",
        "what is this pending charge from you",
        "the tax on the receipt looks incorrect",
        "did my payment on Tuesday go through",
        "why did the subscription price increase",
        "the invoice is missing my company details",
        "I need an itemized bill for March",
        "the currency on my receipt is wrong",
        "clarify the proration on this charge",
    ],
    "cancel": [
        "stop my subscription at the end of the month",
        "I do not want the order anymore, do not ship it",
        "terminate my membership effective today",
        "please cancel the reservation for Saturday",
        "end my free trial before it converts",
        "call off the repair visit, I fixed it myself",
        "cancel the backup plan add-on",
        "I changed my mind about the upgrade, undo it",
        "close my account and stop all charges",
        "withdraw my order for the blue variant",
        "cancel the newsletter, too many emails",
        "abort the pending transfer please",
    ],
    "reschedule": [
        "move my dentist appointment to Friday",
        "push the interview to next week",
        "shift the delivery window to the evening",
        "can we do the call at 3pm instead",
        "postpone the workshop until the afternoon",
        "bring the review meeting forward an hour",
        "swap my Tuesday slot for Thursday",
        "the technician visit needs a new date",
        "delay the onboarding session by two days",
        "change the pickup time to 10am",
        "rebook the demo for Monday morning",
        "move the 1:1 earlier if possible",
    ],
    "techsupport": [
        "the app crashes when I open settings",
        "error 500 when uploading a file",
        "the payment button does nothing when clicked",
        "checkout page freezes at the card step",
        "my invoice PDF will not download",
        "the calendar widget shows the wrong timezone",
        "password reset link never arrives",
        "the cancel button in the app is greyed out",
        "refund status page shows a blank screen",
        "two-factor codes are rejected",
        "sync fails between phone and desktop",
        "the billing tab logs me out every time",
    ],
}


def build_splits(seed: int = 7):
    rng = random.Random(seed)
    examples = []
    for label, texts in _INTENTS.items():
        for t in texts:
            examples.append(dspy.Example(text=t, label=label).with_inputs("text"))
    rng.shuffle(examples)
    # 20 train / 20 val (loop-visible) / 20 holdout (loop-invisible)
    return examples[:20], examples[20:40], examples[40:60]


class Intent(dspy.Signature):
    """Classify the customer message into exactly one intent label."""

    text: str = dspy.InputField()
    label: str = dspy.OutputField(
        desc="one of: refund, billing, cancel, reschedule, techsupport (answer with the bare label)")


def metric(example, pred, trace=None):
    return (getattr(pred, "label", "") or "").strip().lower() == example.label


def evaluate(program, dataset) -> float:
    ok = 0
    for ex in dataset:
        try:
            pred = program(text=ex.text)
            ok += bool(metric(ex, pred))
        except Exception:
            pass
    return ok / len(dataset)


def main() -> None:
    t0 = time.time()
    lm = dspy.LM("openai/gpt-3.5-turbo", temperature=0.0, max_tokens=100)
    dspy.configure(lm=lm)

    train, val, holdout = build_splits()
    program = dspy.Predict(Intent)

    # ── verified channel, BEFORE: baseline on held-out (loop has not run) ──────
    base_holdout = evaluate(program, holdout)
    base_val = evaluate(program, val)

    # ── the EXISTING loop, untouched: GEPA reflective prompt evolution ─────────
    gepa = dspy.GEPA(
        metric=lambda ex, pred, trace=None, pred_name=None, pred_trace=None: metric(ex, pred),
        max_metric_calls=250,                     # hard cost cap
        reflection_lm=dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=2000),
        track_stats=True,
    )
    optimized = gepa.compile(program, trainset=train, valset=val)

    # ── claim channel: what the loop itself reports it achieved ────────────────
    claimed_val = None
    dr = getattr(optimized, "detailed_results", None)
    if dr is not None:
        agg = getattr(dr, "val_aggregate_scores", None)
        if agg:
            claimed_val = max(agg)
    if claimed_val is None:                       # fallback: loop-visible split
        claimed_val = evaluate(optimized, val)
    claimed_delta = claimed_val - base_val

    # ── verified channel, AFTER: independent re-execution on held-out ──────────
    verified_holdout = evaluate(optimized, holdout)
    verified_delta = verified_holdout - base_holdout

    report = {
        "loop": "dspy.GEPA (reflective prompt evolution)",
        "task": "intent classification (synthetic, 5 confusable classes, 20/20/20 split)",
        "model": "task=gpt-3.5-turbo, reflection=gpt-4o-mini",
        "claim_channel": {
            "baseline_val": round(base_val, 3),
            "claimed_best_val": round(claimed_val, 3),
            "claimed_improvement": round(claimed_delta, 3),
        },
        "verified_channel": {
            "baseline_holdout": round(base_holdout, 3),
            "optimized_holdout": round(verified_holdout, 3),
            "verified_improvement": round(verified_delta, 3),
        },
        "audit_gap_pp": round(100 * (claimed_delta - verified_delta), 1),
        "wall_s": round(time.time() - t0, 1),
    }
    _OUT.mkdir(exist_ok=True)
    out = _OUT / "loop_audit_dspy.json"
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    # expects OPENAI_API_KEY in env
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    main()
