"""Two-stage discovery loop.

Stage 1 (discovery): exact DP optimisation against the proxy reward
surfaces the best achievable exploit candidate for each spec.
Stage 2 (verification): the candidate is replayed exactly. It counts as
a flagged exploit only if
  (a) the replayed return matches the DP value exactly (deterministic
      verification, no sampling noise),
  (b) it beats the scripted intended behaviour by a meaningful margin,
  (c) its trajectory shows a generic degeneracy signal (state looping,
      single-action dominance, or non-termination), so a merely better
      honest path does not trip the alarm.

The fuzzer never reads the ground-truth label or exploit-type fields.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import optimizer
from mdp import Spec, intended_policy, rollout

ABS_MARGIN = 10.0   # exploit must beat intended by at least this much
REL_MARGIN = 0.25   # ... or by 25% of |intended return|, whichever is larger


@dataclass
class FuzzResult:
    spec_id: str
    optimal_return: float
    intended_return: float
    diff: float
    margin_threshold: float
    flagged: bool
    verified: bool
    loop_ratio: float
    top_action_share: float
    terminated: bool
    optimal_actions: list = field(default_factory=list)
    seconds: float = 0.0


def degeneracy(traj) -> tuple:
    """Generic, spec-blind degeneracy signals from a trajectory."""
    states = traj.states
    actions = traj.actions
    if not actions:
        return 0.0, 0.0, True
    distinct = len(set(states))
    loop_ratio = 1.0 - distinct / max(1, len(states))
    counts: dict = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    top_share = max(counts.values()) / len(actions)
    return loop_ratio, top_share, traj.terminated


def is_degenerate(loop_ratio: float, top_share: float) -> bool:
    return loop_ratio >= 0.4 or top_share >= 0.6


def fuzz_spec(spec: Spec) -> FuzzResult:
    start = time.perf_counter()
    values, policy, _states = optimizer.solve(spec)
    dp_value = optimizer.optimal_value(spec, values)
    optimal_traj = optimizer.optimal_trajectory(spec, policy)
    intended_traj = rollout(spec, intended_policy(spec))

    loop_ratio, top_share, terminated = degeneracy(optimal_traj)
    opt_ret = optimal_traj.total_return
    int_ret = intended_traj.total_return
    # Exact replay verification: DP prediction must match replay to 1e-6,
    # and a second replay of the intended policy must be identical.
    intended_traj2 = rollout(spec, intended_policy(spec))
    verified = (
        abs(opt_ret - dp_value) < 1e-6
        and abs(intended_traj2.total_return - int_ret) < 1e-9
    )
    diff = opt_ret - int_ret
    threshold = max(ABS_MARGIN, REL_MARGIN * abs(int_ret))
    flagged = bool(
        verified
        and diff > threshold
        and is_degenerate(loop_ratio, top_share)
    )
    return FuzzResult(
        spec_id=spec.id,
        optimal_return=opt_ret,
        intended_return=int_ret,
        diff=diff,
        margin_threshold=threshold,
        flagged=flagged,
        verified=verified,
        loop_ratio=loop_ratio,
        top_action_share=top_share,
        terminated=terminated,
        optimal_actions=list(optimal_traj.actions),
        seconds=time.perf_counter() - start,
    )
