"""Sampled-search audits for small model-free stochastic envs.

Discovery is coordinate-ascent hill climbing over a state->action
policy table, starting from the intended policy. Candidates are
scored on a common seed set (paired comparisons). Verification is
separate: the winning policy and the intended policy are re-scored
on held-out seeds, and the report carries confidence intervals.

Verdicts in this mode are probabilistic. The v0.2 spike measured why:
paired-seed estimates ran about 0.03 optimistic versus fresh seeds,
and a return gap in a stochastic env partly measures baseline
competence. The flag rule therefore also requires degenerate
behaviour in the candidate's held-out rollouts, exactly as in
expected mode.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Hashable

from .audit import REL_INTENDED, REL_RANGE
from .stochastic import behaviour_verdict

Obs = Hashable


class Simulator:
    """Episodic simulator protocol: reset(seed) / step(action)."""

    def __init__(self, reset_fn, step_fn, actions, horizon, id="sim"):
        self._reset = reset_fn
        self._step = step_fn
        self.actions = list(actions)
        self.horizon = horizon
        self.id = id

    def reset(self, seed):
        return self._reset(seed)

    def step(self, action):
        return self._step(action)


def _run_episode(sim: Simulator, act_fn, seed):
    obs = sim.reset(seed)
    states = [obs]
    actions = []
    total = 0.0
    done = False
    steps = 0
    while not done and steps < sim.horizon:
        action = act_fn(obs)
        obs, reward, done = sim.step(action)
        total += float(reward)
        actions.append(action)
        states.append(obs)
        steps += 1
    return total, states, actions, done


def evaluate(sim, act_fn, seeds, collect_states=None):
    returns = []
    for seed in seeds:
        total, states, _a, _d = _run_episode(sim, act_fn, seed)
        returns.append(total)
        if collect_states is not None:
            collect_states.update(states)
    mean = statistics.mean(returns)
    se = (statistics.stdev(returns) / math.sqrt(len(returns))
          if len(returns) > 1 else 0.0)
    return mean, se, returns


@dataclass
class SampledResult:
    spec_id: str
    mode: str = "sampled"
    intended_mean: float = 0.0
    candidate_mean: float = 0.0
    gap: float = 0.0
    gap_ci_low: float = 0.0
    gap_ci_high: float = 0.0
    margin: float = 0.0
    flagged: bool = False
    degenerate: bool = False
    rollout_stats: dict = field(default_factory=dict)
    episodes_spent: int = 0
    seconds: float = 0.0


def audit_sampled(sim: Simulator, intended_fn: Callable,
                  search_episodes=200, budget=60_000,
                  verify_episodes=2000, seed0=20_000) -> SampledResult:
    start = time.perf_counter()
    search_seeds = [seed0 + i for i in range(search_episodes)]
    policy: dict = {}

    def act_fn(obs):
        return policy.get(obs, intended_fn(obs))

    visited: set = set()
    best, _se, _ = evaluate(sim, act_fn, search_seeds, visited)
    worst = best
    spent = search_episodes

    rng = random.Random(0)
    improved = True
    while improved and spent < budget:
        improved = False
        order = list(visited)
        rng.shuffle(order)
        for state in order:
            if spent >= budget:
                break
            current = policy.get(state, intended_fn(state))
            for action in sim.actions:
                if action == current:
                    continue
                policy[state] = action
                value, _se2, _ = evaluate(sim, act_fn, search_seeds,
                                          visited)
                spent += search_episodes
                worst = min(worst, value)
                if value > best:
                    best = value
                    current = action
                    improved = True
                else:
                    policy[state] = current
            policy[state] = current

    # Held-out verification on fresh seeds.
    verify_seeds = [seed0 + 500_000 + i for i in range(verify_episodes)]
    cand_mean, cand_se, _ = evaluate(sim, act_fn, verify_seeds)
    int_mean, int_se, _ = evaluate(sim, intended_fn, verify_seeds)
    spent += 2 * verify_episodes

    # Behaviour of candidate and intended on held-out episodes.
    def behaviour(fn):
        tops, terms, pairs, distincts = [], [], [], []
        for seed in verify_seeds[:400]:
            _t, states, actions, done = _run_episode(sim, fn, seed)
            if actions:
                counts: dict = {}
                pair_counts: dict = {}
                for s_, a in zip(states, actions):
                    counts[a] = counts.get(a, 0) + 1
                    pair_counts[(s_, a)] = pair_counts.get((s_, a), 0) + 1
                tops.append(max(counts.values()) / len(actions))
                pairs.append(max(pair_counts.values()) / len(actions))
                distincts.append(len(set(states)) / len(states))
            terms.append(1.0 if done else 0.0)
        return {
            "mean_top_action_share": (statistics.mean(tops)
                                      if tops else 0.0),
            "mean_pair_dominance": (statistics.mean(pairs)
                                    if pairs else 0.0),
            "mean_distinct_share": (statistics.mean(distincts)
                                    if distincts else 0.0),
            "termination_rate": (statistics.mean(terms)
                                 if terms else 1.0),
            "episodes": min(400, verify_episodes),
        }

    cand_stats = behaviour(act_fn)
    int_stats = behaviour(intended_fn)
    stats = {"candidate": cand_stats, "intended": int_stats}
    degenerate = behaviour_verdict(cand_stats, int_stats)

    gap = cand_mean - int_mean
    gap_se = math.sqrt(cand_se ** 2 + int_se ** 2)
    range_proxy = max(best - worst, 0.0)
    margin = max(REL_INTENDED * abs(int_mean), REL_RANGE * range_proxy)
    flagged = bool(degenerate and gap > margin
                   and (gap - 1.96 * gap_se) > 0)
    return SampledResult(
        spec_id=sim.id, intended_mean=int_mean, candidate_mean=cand_mean,
        gap=gap, gap_ci_low=gap - 1.96 * gap_se,
        gap_ci_high=gap + 1.96 * gap_se, margin=margin,
        flagged=flagged, degenerate=degenerate, rollout_stats=stats,
        episodes_spent=spent, seconds=time.perf_counter() - start)
