#!/usr/bin/env python3
"""Track 2: black-box sampled search where no transition table exists.

The search sees only a simulator: seeded episodes in slippery
FrozenLake, sampled from the same dynamics the exact DP used in
Track 1 (which is what makes the comparison honest: the exact
optimum, 0.744 expected return, is known ground truth).

Method: coordinate-ascent hill climbing over the state->action
policy table, starting from the intended (BFS) policy. Every
candidate is scored on a common set of episode seeds so comparisons
are paired, not luck. Budget is counted in simulated episodes.

Pre-registered kill criterion: if the search recovers less than 80%
of the exact optimal expected return within 100k simulated episodes,
black-box discovery is too weak to build on. If it recovers >= 90%,
search quality is not the blocker for v0.2.

Run: /tmp/gym-venv/bin/python track2_sampled_search.py
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from track1_expected_dp import bfs_policy, load_table  # noqa: E402

HORIZON = 100
EXACT_OPTIMAL = 0.744
EVAL_EPISODES = 400
BUDGET = 100_000


class Simulator:
    """Black-box episodic simulator over a transition table."""

    def __init__(self, table):
        self.table = table
        self.terminal = {s for s, pa in table.items()
                         if all(all(t[3] for t in pa[a]) for a in pa)}

    def episode(self, policy, rng):
        state = 0
        for _ in range(HORIZON):
            action = policy[state]
            outcomes = self.table[state][action]
            x = rng.random()
            acc = 0.0
            for prob, nxt, reward, done in outcomes:
                acc += prob
                if x <= acc:
                    if done:
                        return reward
                    state = nxt
                    break
            else:
                nxt, reward, done = outcomes[-1][1], outcomes[-1][2], \
                    outcomes[-1][3]
                if done:
                    return reward
                state = nxt
        return 0.0

    def evaluate(self, policy, seeds):
        total = 0.0
        for seed in seeds:
            total += self.episode(policy, random.Random(seed))
        return total / len(seeds)


def main():
    table, desc = load_table()
    sim = Simulator(table)
    decision_states = [s for s in table if s not in sim.terminal]
    policy = dict(bfs_policy(table, desc))

    eval_seeds = [10_000 + i for i in range(EVAL_EPISODES)]
    spent = 0
    best = sim.evaluate(policy, eval_seeds)
    spent += EVAL_EPISODES
    print(f"start (intended policy): {best:.3f} "
          f"[exact optimum {EXACT_OPTIMAL}]")

    rng = random.Random(0)
    sweeps = 0
    improved = True
    while improved and spent < BUDGET:
        improved = False
        sweeps += 1
        order = decision_states[:]
        rng.shuffle(order)
        for state in order:
            if spent >= BUDGET:
                break
            current = policy[state]
            for action in range(4):
                if action == current:
                    continue
                policy[state] = action
                value = sim.evaluate(policy, eval_seeds)
                spent += EVAL_EPISODES
                if spent > BUDGET:
                    break
                if value > best:
                    best = value
                    current = action
                    improved = True
                else:
                    policy[state] = current
            policy[state] = current
    print(f"after {sweeps} sweeps, {spent} episodes: search estimate "
          f"{best:.3f}")

    fresh_seeds = [777_000 + i for i in range(20_000)]
    honest = sim.evaluate(policy, fresh_seeds)
    recovery = honest / EXACT_OPTIMAL
    print(f"fresh-seed evaluation of found policy: {honest:.3f} "
          f"({recovery:.1%} of exact optimum)")
    verdict = ("PASS" if recovery >= 0.90 else
               "KILL" if recovery < 0.80 else "MARGINAL")
    print(f"kill criterion (>=90% pass, <80% kill): {verdict}")

    result = {
        "exact_optimal": EXACT_OPTIMAL,
        "intended_start_estimate": None,
        "search_estimate_paired": best,
        "found_policy_fresh_eval": honest,
        "recovery_fraction": recovery,
        "episodes_spent": spent,
        "sweeps": sweeps,
        "verdict": verdict,
        "found_policy": {str(k): v for k, v in sorted(policy.items())},
    }
    out = os.path.join(os.path.dirname(__file__), "track2_results.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
