#!/usr/bin/env python3
"""Track 1: expected-value DP for stochastic envs with a known model.

FrozenLake-v1 slippery (the default): the transition table P carries
probabilities, so backward induction over expectations gives the
provably optimal *expected* return and policy. Two baselines matter:

  script   the v0.1 intended input, an open-loop action sequence.
           In a stochastic env a script cannot react when the ice
           slides the agent somewhere unintended.
  policy   a closed-loop intended policy (state -> action): here the
           greedy shortest-path policy that always moves toward the
           goal along a hole-avoiding route.

The spike question: how large is the exact expected gap for each
baseline form, and does the calibrated flag rule (25% of intended /
10% of range, degeneracy judged on the policy) still discriminate?

Run: /tmp/gym-venv/bin/python track1_expected_dp.py
"""

from __future__ import annotations

import json
import os
from collections import deque

import gymnasium as gym

HERE = os.path.dirname(__file__)
HORIZON = 100


def load_table(map_name=None):
    kwargs = {"is_slippery": True}
    if map_name:
        kwargs["map_name"] = map_name
    env = gym.make("FrozenLake-v1", **kwargs)
    table = env.unwrapped.P
    desc = [row.decode() if isinstance(row, bytes) else row
            for row in env.unwrapped.desc]
    env.close()
    return table, desc


def terminal_states(table):
    return {s for s, pa in table.items()
            if all(all(t[3] for t in pa[a]) for a in pa)}


def backward(table, horizon, choose):
    """choose(outcomes_by_action, t) -> per-state action selector DP.

    Generic finite-horizon expected DP. `choose` is 'optimal' or a
    function (state, t) -> action index.
    """
    states = list(table.keys())
    V = {s: 0.0 for s in states}
    policy = {}
    terminal = terminal_states(table)
    for t in range(horizon - 1, -1, -1):
        newV = {}
        for s in states:
            if s in terminal:
                newV[s] = 0.0
                continue
            actions = {}
            for a, outcomes in table[s].items():
                q = 0.0
                for prob, nxt, reward, done in outcomes:
                    q += prob * (reward + (0.0 if done else V[nxt]))
                actions[a] = q
            if choose == "optimal":
                a_best = max(actions, key=actions.get)
                policy[(s, t)] = a_best
                newV[s] = actions[a_best]
            else:
                a = choose(s, t)
                newV[s] = actions[a]
        V = newV
    return V, policy


def bfs_policy(table, desc):
    """Closed-loop intended policy: BFS shortest hole-avoiding route,
    expressed as a state->action table (follow the BFS tree)."""
    nrow, ncol = len(desc), len(desc[0])
    goal = nrow * ncol - 1
    holes = {i * ncol + j for i, row in enumerate(desc)
             for j, ch in enumerate(row) if ch == "H"}
    # BFS over the *intended* transition (most likely outcome): use the
    # deterministic skeleton by taking, for each action, the outcome
    # with the highest probability as the nominal next state.
    nominal = {}
    for s, pa in table.items():
        for a, outcomes in pa.items():
            nominal[(s, a)] = max(outcomes, key=lambda o: o[0])[1]
    prev = {0: None}
    queue = deque([0])
    while queue:
        s = queue.popleft()
        if s == goal:
            break
        for a in range(4):
            ns = nominal[(s, a)]
            if ns not in prev and ns not in holes:
                prev[ns] = (s, a)
                queue.append(ns)
    policy = {}
    for s in table:
        if s == goal or s in holes:
            policy[s] = 0
            continue
        # walk from s along BFS parents is wrong direction; instead do
        # a fresh BFS from s to goal for the first action.
        seen = {s: None}
        q = deque([s])
        first = None
        while q:
            cur = q.popleft()
            if cur == goal:
                node = cur
                while seen[node] is not None:
                    node_prev, node_a = seen[node]
                    first = node_a
                    node = node_prev
                break
            for a in range(4):
                ns = nominal[(cur, a)]
                if ns not in seen and ns not in holes:
                    seen[ns] = (cur, a)
                    q.append(ns)
        policy[s] = first if first is not None else 0
    return policy


def main():
    results = {}
    for label, map_name in (("4x4", None), ("8x8", "8x8")):
        table, desc = load_table(map_name)
        ncol = len(desc[0])
        goal = len(desc) * ncol - 1
        start = 0

        V_opt, _pol = backward(table, HORIZON, "optimal")
        optimal = V_opt[start]

        intended_map = bfs_policy(table, desc)
        V_pol, _ = backward(table, HORIZON,
                            lambda s, t: intended_map[s])
        policy_baseline = V_pol[start]

        # Open-loop script: the BFS route's action sequence from start.
        script = []
        s = start
        seen = set()
        while s != goal and s not in seen and len(script) < HORIZON:
            seen.add(s)
            a = intended_map[s]
            script.append(a)
            outcomes = table[s][a]
            s = max(outcomes, key=lambda o: o[0])[1]
        V_script, _ = backward(
            table, HORIZON,
            lambda s, t: script[t] if t < len(script) else 0)
        script_baseline = V_script[start]

        # Pessimal expected value for the range term.
        def worst(s, t):
            return None  # handled below by min variant
        # min-DP inline:
        states = list(table.keys())
        terminal = terminal_states(table)
        V = {s: 0.0 for s in states}
        for t in range(HORIZON - 1, -1, -1):
            newV = {}
            for s in states:
                if s in terminal:
                    newV[s] = 0.0
                    continue
                qs = []
                for a, outcomes in table[s].items():
                    q = sum(p * (r + (0.0 if d else V[n]))
                            for p, n, r, d in outcomes)
                    qs.append(q)
                newV[s] = min(qs)
            V = newV
        pessimal = V[start]
        rng_range = optimal - pessimal

        for base_name, base in (("script", script_baseline),
                                ("policy", policy_baseline)):
            gap = optimal - base
            margin = max(0.25 * abs(base), 0.10 * rng_range)
            results[f"{label}-{base_name}"] = {
                "optimal_expected": optimal,
                "baseline_expected": base,
                "pessimal_expected": pessimal,
                "gap": gap, "margin": margin,
                "gap_clears_margin": gap > margin,
            }
            print(f"FrozenLake {label} slippery, baseline={base_name}: "
                  f"optimal={optimal:.3f} baseline={base:.3f} "
                  f"pessimal={pessimal:.3f} gap={gap:.3f} "
                  f"margin={margin:.3f} clears={gap > margin}")

    out = os.path.join(HERE, "track1_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
