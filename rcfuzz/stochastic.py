"""Stochastic audits: expected-value DP and rollout behaviour.

For envs whose transition table carries probabilities, backward
induction over expectations is still exact: optimal expected return,
intended-policy expected return, and pessimal expected return are
all computable. What changes is the flag rule, per the v0.2 spike:

  * The intended baseline must be a closed-loop policy. An open-loop
    script in a stochastic env measures baseline incompetence, not
    reward design (a script scores 0.001 on slippery FrozenLake,
    whose reward is clean).
  * A return gap alone never flags. Stochastic mode also requires
    degenerate behaviour in the candidate policy's own rollout
    distribution: farming or looping across many seeded episodes, or
    a large non-termination mass.
  * Every stochastic report carries the baseline-competence caveat:
    part of any gap may be control skill, not incentive design.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Hashable

from .audit import REL_INTENDED, REL_RANGE

State = Hashable
Action = str


@dataclass
class ProbSpec:
    """A stochastic MDP with a known transition table.

    outcomes(state, action) -> list of (prob, next_state, reward, done)
    """
    id: str
    description: str
    horizon: int
    initial_state: State
    outcomes_fn: Callable
    actions_fn: Callable
    is_terminal_fn: Callable
    # Closed-loop intended policy: fn(state, t) -> action. Required.
    intended_policy_fn: Callable

    def actions(self, state):
        return list(self.actions_fn(state))

    def outcomes(self, state, action):
        return self.outcomes_fn(state, action)

    def is_terminal(self, state):
        return bool(self.is_terminal_fn(state))


def _reachable(prob: ProbSpec):
    states_by_time = [set() for _ in range(prob.horizon + 1)]
    states_by_time[0].add(prob.initial_state)
    for t in range(prob.horizon):
        for state in states_by_time[t]:
            if prob.is_terminal(state):
                continue
            for action in prob.actions(state):
                for _p, ns, _r, _d in prob.outcomes(state, action):
                    states_by_time[t + 1].add(ns)
    return states_by_time


def _expected_backward(prob, states_by_time, choose):
    """choose: 'optimal', 'pessimal', or fn(state, t) -> action."""
    horizon = prob.horizon
    values = [dict() for _ in range(horizon + 1)]
    policy = [dict() for _ in range(horizon + 1)]
    for state in states_by_time[horizon]:
        values[horizon][state] = 0.0
    for t in range(horizon - 1, -1, -1):
        for state in states_by_time[t]:
            if prob.is_terminal(state):
                values[t][state] = 0.0
                continue
            qs = {}
            for action in prob.actions(state):
                q = 0.0
                for p, ns, reward, done in prob.outcomes(state, action):
                    future = 0.0 if done else values[t + 1].get(ns, 0.0)
                    q += p * (float(reward) + future)
                qs[action] = q
            if choose == "optimal":
                best = max(qs, key=qs.get)
                policy[t][state] = best
                values[t][state] = qs[best]
            elif choose == "pessimal":
                values[t][state] = min(qs.values())
            else:
                values[t][state] = qs[choose(state, t)]
    return values, policy


def expected_optimal(prob: ProbSpec):
    states_by_time = _reachable(prob)
    values, policy = _expected_backward(prob, states_by_time, "optimal")
    return float(values[0].get(prob.initial_state, 0.0)), policy, \
        states_by_time


def expected_evaluate(prob: ProbSpec, policy_fn, states_by_time=None):
    if states_by_time is None:
        states_by_time = _reachable(prob)
    values, _ = _expected_backward(prob, states_by_time, policy_fn)
    return float(values[0].get(prob.initial_state, 0.0))


def expected_pessimal(prob: ProbSpec, states_by_time=None):
    if states_by_time is None:
        states_by_time = _reachable(prob)
    values, _ = _expected_backward(prob, states_by_time, "pessimal")
    return float(values[0].get(prob.initial_state, 0.0))


def sample_outcome(outcomes, rng: random.Random):
    x = rng.random()
    acc = 0.0
    for prob_, ns, reward, done in outcomes:
        acc += prob_
        if x <= acc:
            return ns, reward, done
    _p, ns, reward, done = outcomes[-1]
    return ns, reward, done


def rollout_stats(prob: ProbSpec, policy_table, episodes=400, seed=1234):
    """Behaviour of a (time-dependent) policy across seeded episodes.

    Loop ratios are NOT evidence in stochastic envs (ice makes honest
    policies revisit states), so the stats are the ones that separate
    farming from navigation: termination rate, the share of all
    steps spent on the single most-used (state, action) pair, and
    the share of distinct states per episode.
    """
    rng = random.Random(seed)
    tops, terms, pairs, distincts = [], [], [], []
    for _ in range(episodes):
        state = prob.initial_state
        states = [state]
        actions = []
        pair_counts: dict = {}
        done = prob.is_terminal(state)
        t = 0
        while not done and t < prob.horizon:
            action = policy_table[t].get(state)
            if action is None:
                action = prob.actions(state)[0]
            ns, _r, done = sample_outcome(prob.outcomes(state, action),
                                          rng)
            actions.append(action)
            pair_counts[(state, action)] = \
                pair_counts.get((state, action), 0) + 1
            states.append(ns)
            state = ns
            t += 1
            if prob.is_terminal(state):
                done = True
        if actions:
            counts: dict = {}
            for a in actions:
                counts[a] = counts.get(a, 0) + 1
            tops.append(max(counts.values()) / len(actions))
            pairs.append(max(pair_counts.values()) / len(actions))
            distincts.append(len(set(states)) / len(states))
        terms.append(1.0 if done else 0.0)
    return {
        "mean_top_action_share": statistics.mean(tops) if tops else 0.0,
        "mean_pair_dominance": statistics.mean(pairs) if pairs else 0.0,
        "mean_distinct_share": (statistics.mean(distincts)
                                if distincts else 0.0),
        "termination_rate": statistics.mean(terms) if terms else 1.0,
        "episodes": episodes,
    }


def behaviour_verdict(candidate_stats, intended_stats) -> bool:
    """Degeneracy in stochastic mode is relative to the intended
    policy's behaviour in the same stochastic env. Any of:

      * termination drop >= 0.30 (refuses to finish where the
        intended policy finishes),
      * one (state, action) pair takes >= 50% of all steps and at
        least twice the intended policy's pair dominance (farming),
      * the candidate lives in a tiny state set: distinct-state
        share <= 0.15 and at most half the intended policy's.
    """
    term_drop = (intended_stats["termination_rate"]
                 - candidate_stats["termination_rate"])
    if term_drop >= 0.30:
        return True
    pair = candidate_stats["mean_pair_dominance"]
    if pair >= 0.50 and pair >= 2.0 * max(
            intended_stats["mean_pair_dominance"], 1e-9):
        return True
    distinct = candidate_stats["mean_distinct_share"]
    if distinct <= 0.15 and distinct <= 0.5 * \
            intended_stats["mean_distinct_share"]:
        return True
    return False


@dataclass
class StochasticResult:
    spec_id: str
    mode: str  # "expected"
    intended_expected: float
    optimal_expected: float
    pessimal_expected: float
    return_range: float
    gap: float
    margin: float
    flagged: bool
    degenerate: bool
    rollout_stats: dict = field(default_factory=dict)
    seconds: float = 0.0


def audit_stochastic(prob: ProbSpec, episodes=400) -> StochasticResult:
    start = time.perf_counter()
    optimal, policy, states_by_time = expected_optimal(prob)
    intended = expected_evaluate(prob, prob.intended_policy_fn,
                                 states_by_time)
    pessimal = expected_pessimal(prob, states_by_time)
    stats = rollout_stats(prob, policy, episodes=episodes)
    all_states = set()
    for layer in states_by_time:
        all_states.update(layer)
    intended_table = [
        {s: prob.intended_policy_fn(s, t) for s in all_states}
        for t in range(prob.horizon + 1)]
    intended_stats = rollout_stats(prob, intended_table,
                                   episodes=episodes, seed=4321)
    degenerate = behaviour_verdict(stats, intended_stats)
    stats = {"candidate": stats, "intended": intended_stats}
    gap = optimal - intended
    return_range = optimal - pessimal
    margin = max(REL_INTENDED * abs(intended), REL_RANGE * return_range)
    flagged = bool(gap > margin and degenerate)
    return StochasticResult(
        spec_id=prob.id, mode="expected",
        intended_expected=intended, optimal_expected=optimal,
        pessimal_expected=pessimal, return_range=return_range,
        gap=gap, margin=margin, flagged=flagged, degenerate=degenerate,
        rollout_stats=stats, seconds=time.perf_counter() - start)
