"""The audit: optimizer as auditor, proof by replay.

Stage 1 (discovery): exact DP optimisation against the stated reward
surfaces the best achievable candidate policy.
Stage 2 (verification): the candidate is replayed exactly. It counts
as an exploit only if
  (a) the replayed return matches the DP value (deterministic
      verification, no sampling noise),
  (b) it beats the scripted intended behaviour by a scale-aware
      margin: more than max(REL_INTENDED of |intended return|,
      REL_RANGE of the return range), where the return range is
      optimal minus pessimal return, and
  (c) its trajectory shows a generic degeneracy signal (state looping
      or single-action dominance), so a merely better honest path
      does not trip the alarm.

The margin was calibrated on a 20-spec corpus (spike + borderline):
10/10 planted exploits, 0/10 false alarms. See validation/VALIDATION.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import optimizer
from .mdp import Spec, intended_policy, rollout

REL_INTENDED = 0.25   # gap must exceed 25% of |intended return| ...
REL_RANGE = 0.10      # ... or 10% of the return range, whichever larger
LOOP_RATIO_MIN = 0.4
TOP_ACTION_MIN = 0.6


@dataclass
class AuditResult:
    spec_id: str
    intended_return: float
    optimal_return: float
    pessimal_return: float
    return_range: float
    gap: float
    margin: float
    flagged: bool
    verified: bool
    degenerate: bool
    loop_ratio: float
    top_action_share: float
    terminated: bool
    horizon: int
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
    return loop_ratio >= LOOP_RATIO_MIN or top_share >= TOP_ACTION_MIN


def audit_spec(spec: Spec) -> AuditResult:
    start = time.perf_counter()
    values, policy, states_by_time = optimizer.solve(spec)
    dp_value = optimizer.optimal_value(spec, values)
    optimal_traj = optimizer.optimal_trajectory(spec, policy)
    intended_traj = rollout(spec, intended_policy(spec))
    intended_traj2 = rollout(spec, intended_policy(spec))
    pessimal = optimizer.pessimal_value(spec, states_by_time)

    loop_ratio, top_share, terminated = degeneracy(optimal_traj)
    opt_ret = optimal_traj.total_return
    int_ret = intended_traj.total_return
    verified = (
        abs(opt_ret - dp_value) < 1e-6
        and abs(intended_traj2.total_return - int_ret) < 1e-9
    )
    gap = opt_ret - int_ret
    return_range = opt_ret - pessimal
    margin = max(REL_INTENDED * abs(int_ret), REL_RANGE * return_range)
    degenerate = is_degenerate(loop_ratio, top_share)
    flagged = bool(verified and gap > margin and degenerate)
    return AuditResult(
        spec_id=spec.id,
        intended_return=int_ret,
        optimal_return=opt_ret,
        pessimal_return=pessimal,
        return_range=return_range,
        gap=gap,
        margin=margin,
        flagged=flagged,
        verified=verified,
        degenerate=degenerate,
        loop_ratio=loop_ratio,
        top_action_share=top_share,
        terminated=terminated,
        horizon=spec.horizon,
        optimal_actions=list(optimal_traj.actions),
        seconds=time.perf_counter() - start,
    )
