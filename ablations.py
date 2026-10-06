#!/usr/bin/env python3
"""Ablations for the spike report.

1. Margin sweep: would other margin thresholds still separate?
2. Degeneracy alone: does the degeneracy signal separate by itself?
3. Random baseline: how well does random policy search do vs exact DP?

Usage: python3 ablations.py
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, "src")

from corpus import build_corpus  # noqa: E402
from fuzzer import degeneracy, fuzz_spec, is_degenerate  # noqa: E402
from mdp import rollout  # noqa: E402


def random_best_return(spec, trials=300, seed=0):
    rng = random.Random(seed)
    actions_cache = {}

    def rand_policy(state, t):
        if state not in actions_cache:
            actions_cache[state] = spec.actions(state)
        return rng.choice(actions_cache[state])

    best = None
    for _ in range(trials):
        ret = rollout(spec, rand_policy).total_return
        if best is None or ret > best:
            best = ret
    return best


def main():
    corpus = build_corpus()
    results = {s.id: fuzz_spec(s) for s in corpus}

    print("== Margin sweep (flag = diff > max(abs, 25%|intended|) + degeneracy) ==")
    for abs_margin in (1.0, 5.0, 10.0, 20.0, 30.0):
        hits = sum(
            1 for s in corpus if s.label == "exploitable"
            and results[s.id].diff > max(abs_margin, 0.25 * abs(results[s.id].intended_return))
            and is_degenerate(results[s.id].loop_ratio, results[s.id].top_action_share)
        )
        fas = sum(
            1 for s in corpus if s.label == "clean"
            and results[s.id].diff > max(abs_margin, 0.25 * abs(results[s.id].intended_return))
            and is_degenerate(results[s.id].loop_ratio, results[s.id].top_action_share)
        )
        print(f"  abs margin {abs_margin:>5.1f}: hits {hits}/6, false alarms {fas}/6")

    print("\n== Degeneracy signal alone (no margin) ==")
    for s in corpus:
        r = results[s.id]
        deg = is_degenerate(r.loop_ratio, r.top_action_share)
        print(f"  {s.id:<20} label={s.label:<12} degenerate={deg}")

    print("\n== Random policy search (300 rollouts/spec) vs exact DP ==")
    for s in corpus:
        rb = random_best_return(s)
        r = results[s.id]
        print(f"  {s.id:<20} label={s.label:<12} random_best={rb:>7.1f} "
              f"dp_optimal={r.optimal_return:>7.1f} intended={r.intended_return:>6.1f}")


if __name__ == "__main__":
    main()
