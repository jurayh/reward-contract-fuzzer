#!/usr/bin/env python3
"""Run the Reward Contract Fuzzer on real third-party Gymnasium envs.

Environments (gymnasium 1.3.0, installed from PyPI, not our code):
  FrozenLake-v1 (is_slippery=False), Taxi-v4, CliffWalking-v1.

The adapter reads each env's published transition table `P` — the same
object any Gymnasium user gets — and builds the spike's MDP Spec from
it. Nothing about the envs is reimplemented. Intended behaviour is a
scripted policy per env, defined the way the env's own documentation
describes the task:

  FrozenLake   : shortest hole-avoiding path to the goal
  Taxi         : greedy drive-pickup-deliver from a fixed start state
  CliffWalking : the SAFE route (away from the cliff), which the env
                 documents as intended; the return-optimal route hugs
                 the cliff edge. This is the interesting case: a better
                 honest path is not an exploit, and the flag rule must
                 not fire on it.

Run with the gymnasium venv: /tmp/gym-venv/bin/python run_gym_fuzzer.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import gymnasium as gym  # noqa: E402

from fuzzer import fuzz_spec  # noqa: E402
from mdp import Spec  # noqa: E402

ACTION_NAMES = {
    "FrozenLake-v1": ["L", "D", "R", "U"],
    "Taxi-v4": ["S", "N", "E", "W", "pickup", "dropoff"],
    "CliffWalking-v1": ["up", "right", "down", "left"],
}
HORIZONS = {"FrozenLake-v1": 16, "Taxi-v4": 40, "CliffWalking-v1": 25}


def build_spec(env_id, initial_state, intended_actions):
    env = gym.make(env_id, is_slippery=False) if env_id == "FrozenLake-v1" \
        else gym.make(env_id)
    unwrapped = env.unwrapped
    table = unwrapped.P
    names = ACTION_NAMES[env_id]
    n_actions = len(names)

    terminal = set()
    for state, per_action in table.items():
        if all(all(t[3] for t in per_action[a]) for a in per_action):
            terminal.add(state)

    def step_fn(state, action):
        outcomes = table[state][names.index(action)]
        assert len(outcomes) == 1 and outcomes[0][0] == 1.0, "stochastic env"
        _p, nxt, reward, done = outcomes[0]
        return nxt, float(reward), bool(done)

    spec = Spec(
        id=env_id,
        description=f"real Gymnasium env {env_id}",
        horizon=HORIZONS[env_id],
        initial_state=initial_state,
        step_fn=step_fn,
        actions_fn=lambda s: list(names),
        is_terminal_fn=lambda s: s in terminal,
        intended_actions=intended_actions,
        label="unknown",
    )
    env.close()
    return spec


def bfs_script(env_id, start, goal_pred, banned_pred=lambda s: False):
    """Shortest action script from start to any goal state (deterministic)."""
    env = gym.make(env_id, is_slippery=False) if env_id == "FrozenLake-v1" \
        else gym.make(env_id)
    table = env.unwrapped.P
    names = ACTION_NAMES[env_id]
    prev = {start: None}
    queue = deque([start])
    goal = None
    while queue:
        state = queue.popleft()
        if goal_pred(state):
            goal = state
            break
        for a_idx, outcomes in table[state].items():
            _p, nxt, _r, done = outcomes[0]
            if nxt not in prev and not banned_pred(nxt):
                prev[nxt] = (state, names[a_idx])
                queue.append(nxt)
    assert goal is not None, f"no path for {env_id}"
    script = []
    node = goal
    while prev[node] is not None:
        state, action = prev[node]
        script.append(action)
        node = state
    env.close()
    return list(reversed(script))


def taxi_script_via_dp(start):
    """Script that follows the spike optimizer's own optimal policy.

    Only used to define the intended baseline for Taxi (drive to the
    passenger, pick up, deliver). If the intended script is optimal,
    the fuzzer must report a gap of 0.
    """
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    import optimizer
    spec = build_spec("Taxi-v4", start, [])
    _values, policy, _states = optimizer.solve(spec)
    script, state = [], start
    for _t in range(spec.horizon):
        action = policy[_t][state]
        script.append(action)
        state, _r, done = spec.step(state, action)
        if done:
            break
    return script


def main():
    results = {}

    frozen = build_spec(
        "FrozenLake-v1", 0,
        bfs_script("FrozenLake-v1", 0, lambda s: s == 15,
                   banned_pred=lambda s: s in (5, 7, 11, 12)))
    cliff_safe = ["up", "up"] + ["right"] * 11 + ["down", "down"]
    cliff = build_spec("CliffWalking-v1", 36, cliff_safe)
    taxi_start = 328  # fixed deterministic start state in Taxi's encoding
    taxi = build_spec("Taxi-v4", taxi_start, taxi_script_via_dp(taxi_start))

    for spec in (frozen, taxi, cliff):
        r = fuzz_spec(spec)
        results[spec.id] = {
            "intended_return": r.intended_return,
            "optimal_return": r.optimal_return,
            "diff": r.diff,
            "margin_threshold": r.margin_threshold,
            "flagged": r.flagged,
            "verified": r.verified,
            "terminated": r.terminated,
            "loop_ratio": r.loop_ratio,
            "top_action_share": r.top_action_share,
            "seconds": r.seconds,
        }
        print(f"{spec.id:<16} intended={r.intended_return:>6.1f} "
              f"optimal={r.optimal_return:>6.1f} diff={r.diff:>6.1f} "
              f"flagged={r.flagged} verified={r.verified}")

    out = os.path.join(HERE, "gym_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
