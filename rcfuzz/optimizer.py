"""Exact deterministic optimisation over an MDP Spec.

Finite-horizon dynamic programming (backward induction) over the states
reachable from the initial state. The returned policy is provably
optimal within the horizon, so a miss cannot be blamed on a weak
optimiser, and a hit is a real property of the reward spec.
"""

from __future__ import annotations

from .mdp import Spec, Trajectory, rollout


def enumerate_reachable(spec: Spec) -> list:
    """Forward BFS: states_by_time[t] = set of states reachable in t steps."""
    states_by_time: list = [set() for _ in range(spec.horizon + 1)]
    states_by_time[0].add(spec.initial_state)
    for t in range(spec.horizon):
        for state in states_by_time[t]:
            if spec.is_terminal(state):
                continue
            for action in spec.actions(state):
                ns, _r, _done = spec.step(state, action)
                states_by_time[t + 1].add(ns)
    return states_by_time


def _backward(spec: Spec, states_by_time, maximise: bool):
    horizon = spec.horizon
    values: list = [dict() for _ in range(horizon + 1)]
    policy: list = [dict() for _ in range(horizon + 1)]
    for state in states_by_time[horizon]:
        values[horizon][state] = 0.0
    for t in range(horizon - 1, -1, -1):
        for state in states_by_time[t]:
            if spec.is_terminal(state):
                values[t][state] = 0.0
                continue
            best_value = None
            best_action = None
            for action in spec.actions(state):
                ns, reward, done = spec.step(state, action)
                future = 0.0 if done else values[t + 1].get(ns, 0.0)
                q = float(reward) + future
                if best_value is None or (q > best_value if maximise
                                          else q < best_value):
                    best_value = q
                    best_action = action
            values[t][state] = float(best_value)
            policy[t][state] = best_action
    return values, policy


def solve(spec: Spec):
    """Return (value_table, policy_table, states_by_time) for the optimum."""
    states_by_time = enumerate_reachable(spec)
    values, policy = _backward(spec, states_by_time, maximise=True)
    return values, policy, states_by_time


def pessimal_value(spec: Spec, states_by_time=None) -> float:
    """Worst achievable return from the initial state (exact, by DP)."""
    if states_by_time is None:
        states_by_time = enumerate_reachable(spec)
    values, _policy = _backward(spec, states_by_time, maximise=False)
    return float(values[0].get(spec.initial_state, 0.0))


def optimal_trajectory(spec: Spec, policy: list) -> Trajectory:
    def pi(state, t):
        action = policy[t].get(state)
        if action is None:
            return spec.actions(state)[0]
        return action

    return rollout(spec, pi)


def optimal_value(spec: Spec, values: list) -> float:
    return float(values[0].get(spec.initial_state, 0.0))
