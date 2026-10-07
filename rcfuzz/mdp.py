"""Minimal deterministic MDP protocol for Reward Contract Fuzzer.

A Spec exposes an MDP as a black box: initial state, available actions,
a deterministic step function, and a termination predicate. The auditor
only ever uses this protocol plus a scripted intended policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable

State = Hashable
Action = str


@dataclass
class Trajectory:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return float(sum(self.rewards))

    @property
    def terminated(self) -> bool:
        return getattr(self, "_terminated", False)


@dataclass
class Spec:
    id: str
    description: str
    horizon: int
    initial_state: State
    # step(state, action) -> (next_state, reward, done)
    step_fn: Callable[[State, Action], tuple]
    actions_fn: Callable[[State], list]
    is_terminal_fn: Callable[[State], bool]
    intended_actions: list  # scripted intended behaviour (action sequence)
    # Optional closed-loop intended policy: fn(state, t) -> action.
    # When set it takes precedence over the script. Required for any
    # meaningful baseline in stochastic envs; accepted here so a
    # deterministic audit can also use policy-form baselines.
    intended_policy_fn: Callable | None = None

    def actions(self, state: State) -> list:
        return list(self.actions_fn(state))

    def step(self, state: State, action: Action):
        return self.step_fn(state, action)

    def is_terminal(self, state: State) -> bool:
        return bool(self.is_terminal_fn(state))


def rollout(spec: Spec, policy: Callable[[State, int], Action]) -> Trajectory:
    """Replay a (possibly time-dependent) policy from the initial state."""
    traj = Trajectory()
    state = spec.initial_state
    traj.states.append(state)
    done = spec.is_terminal(state)
    t = 0
    while not done and t < spec.horizon:
        action = policy(state, t)
        next_state, reward, done = spec.step(state, action)
        traj.actions.append(action)
        traj.rewards.append(float(reward))
        traj.states.append(next_state)
        state = next_state
        t += 1
        if spec.is_terminal(state):
            done = True
    traj._terminated = done  # type: ignore[attr-defined]
    return traj


def intended_policy(spec: Spec) -> Callable[[State, int], Action]:
    """Intended behaviour: the closed-loop policy if given, else the
    script followed by a harmless idle."""
    if spec.intended_policy_fn is not None:
        return spec.intended_policy_fn
    script = list(spec.intended_actions)

    def policy(state: State, t: int) -> Action:
        if t < len(script):
            return script[t]
        available = spec.actions(state)
        for idle in ("idle", "stay", "wait"):
            if idle in available:
                return idle
        return available[0]

    return policy
