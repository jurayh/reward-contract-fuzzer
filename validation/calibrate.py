#!/usr/bin/env python3
"""Margin calibration: old rule vs scale-aware candidates.

For every spec (spike corpus, borderline corpus) this computes the
intended return, the optimal return, and the pessimal (worst-policy)
return, then scores four flag rules:

  R0 spike rule     diff > max(10, 25% of |intended|) and degenerate
  R1 relative       diff > 25% of |intended| and degenerate
  R2 range-aware    diff > max(25% of |intended|, 10% of return range)
                    and degenerate
  R3 range-only     diff > 15% of return range and degenerate

Return range = optimal - pessimal, both exact by DP. Degeneracy is
the spike's generic signal (loop ratio >= 0.4 or top action share
>= 0.6 on the optimal trajectory). All rules keep the degeneracy
gate; the question is only which margin travels across reward scales.

Usage: python3 calibrate.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import optimizer  # noqa: E402
from borderline import borderline_corpus  # noqa: E402
from corpus import build_corpus as spike_corpus  # noqa: E402
from fuzzer import degeneracy  # noqa: E402
from mdp import intended_policy, rollout  # noqa: E402


def pessimal_value(spec) -> float:
    states_by_time = optimizer.enumerate_reachable(spec)
    horizon = spec.horizon
    values = [dict() for _ in range(horizon + 1)]
    for state in states_by_time[horizon]:
        values[horizon][state] = 0.0
    for t in range(horizon - 1, -1, -1):
        for state in states_by_time[t]:
            if spec.is_terminal(state):
                values[t][state] = 0.0
                continue
            worst = None
            for action in spec.actions(state):
                ns, reward, done = spec.step(state, action)
                future = 0.0 if done else values[t + 1].get(ns, 0.0)
                q = float(reward) + future
                if worst is None or q < worst:
                    worst = q
            values[t][state] = float(worst)
    return float(values[0].get(spec.initial_state, 0.0))


def measure(spec):
    values, policy, _ = optimizer.solve(spec)
    optimal_traj = optimizer.optimal_trajectory(spec, policy)
    intended_traj = rollout(spec, intended_policy(spec))
    loop_ratio, top_share, _term = degeneracy(optimal_traj)
    degenerate = loop_ratio >= 0.4 or top_share >= 0.6
    opt = optimal_traj.total_return
    intended = intended_traj.total_return
    pess = pessimal_value(spec)
    return {
        "id": spec.id, "label": spec.label,
        "intended": intended, "optimal": opt, "pessimal": pess,
        "range": opt - pess, "diff": opt - intended,
        "degenerate": degenerate,
    }


RULES = {
    "R0 spike (abs 10 / 25%)":
        lambda m: m["diff"] > max(10.0, 0.25 * abs(m["intended"])),
    "R1 relative 25%":
        lambda m: m["diff"] > 0.25 * abs(m["intended"]),
    "R2 range-aware (25% / 10% range)":
        lambda m: m["diff"] > max(0.25 * abs(m["intended"]),
                                  0.10 * m["range"]),
    "R3 range-only 15%":
        lambda m: m["diff"] > 0.15 * m["range"],
}


def score(rows, suite_name):
    print(f"\n== {suite_name} ==")
    header = (f"{'spec':<20} {'label':<12} {'int':>9} {'opt':>9} "
              f"{'diff':>9} {'range':>9} {'degen':>5}  "
              + "  ".join(f"{k.split()[0]:>3}" for k in RULES))
    print(header)
    stats = {k: [0, 0, 0, 0] for k in RULES}  # tp, fp, tn, fn
    for m in rows:
        flags = {}
        for name, rule in RULES.items():
            flags[name] = bool(rule(m) and m["degenerate"])
            truth = m["label"] == "exploitable"
            idx = (0 if flags[name] and truth else
                   1 if flags[name] else
                   2 if truth else 3)
            # tp fp fn tn mapping
            if flags[name] and truth:
                stats[name][0] += 1
            elif flags[name]:
                stats[name][1] += 1
            elif truth:
                stats[name][2] += 1
            else:
                stats[name][3] += 1
        print(f"{m['id']:<20} {m['label']:<12} {m['intended']:>9.2f} "
              f"{m['optimal']:>9.2f} {m['diff']:>9.2f} {m['range']:>9.2f} "
              f"{str(m['degenerate']):>5}  "
              + "  ".join(("  Y" if flags[k] else "  .") for k in RULES))
    for name in RULES:
        tp, fp, fn, tn = stats[name]
        print(f"{name:<34} hits {tp}/{tp + fn}  false alarms {fp}/{fp + tn}")
    return stats


def main():
    spike_rows = [measure(s) for s in spike_corpus()]
    border_rows = [measure(s) for s in borderline_corpus()]
    all_rows = spike_rows + border_rows
    score(spike_rows, "spike corpus (regression)")
    score(border_rows, "borderline corpus")
    score(all_rows, "combined")
    out = os.path.join(HERE, "calibration.json")
    with open(out, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print("\nWrote", out)


if __name__ == "__main__":
    main()
