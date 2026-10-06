"""Environment adapters: turn real envs into MDP Specs.

Two adapters ship in v0.1:

  gymnasium  Envs that expose a deterministic transition table `P`
             (FrozenLake, Taxi, CliffWalking, and friends). Stochastic
             tables are rejected with a clear error: exact DP would
             silently misreport them.
  stateful   A Python module defining an Env class with
             reset/step/get_state/set_state plus module-level ACTIONS
             (and optionally INTENDED, HORIZON), the convention used
             by the project's example envs. Terminal states are
             discovered by probing transitions, bounded by horizon
             and max_states.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from .mdp import Spec


class AuditNotApplicable(Exception):
    """The env cannot be audited exactly by this tool (yet)."""


def gymnasium_spec(env_id, intended_actions, horizon=None,
                   kwargs=None, action_names=None, max_states=200_000):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise AuditNotApplicable(
            "gymnasium is not installed; install the 'gymnasium' extra"
        ) from exc
    env = gym.make(env_id, **(kwargs or {}))
    unwrapped = env.unwrapped
    table = getattr(unwrapped, "P", None)
    if table is None:
        env.close()
        raise AuditNotApplicable(
            f"{env_id} does not expose a transition table P; v0.1 audits "
            "enumerable deterministic envs only"
        )
    if len(table) > max_states:
        env.close()
        raise AuditNotApplicable(
            f"{env_id} has {len(table)} states, over the {max_states} cap"
        )
    n_actions = len(next(iter(table.values())))
    names = action_names or [str(i) for i in range(n_actions)]
    if len(names) != n_actions:
        env.close()
        raise AuditNotApplicable("action_names length does not match env")

    terminal = set()
    for state, per_action in table.items():
        for a_idx, outcomes in per_action.items():
            for prob, _nxt, _r, _d in outcomes:
                if prob not in (0.0, 1.0):
                    env.close()
                    raise AuditNotApplicable(
                        f"{env_id} is stochastic (P has probabilities "
                        "other than 0/1); exact DP would misreport it. "
                        "Sampled search is not in v0.1."
                    )
        if all(all(t[3] for t in per_action[a]) for a in per_action):
            terminal.add(state)

    obs, _info = env.reset(seed=0)
    initial = int(obs)
    if horizon is None:
        horizon = getattr(env.spec, "max_episode_steps", None) or 100

    def step_fn(state, action):
        outcomes = table[state][names.index(action)]
        _p, nxt, reward, done = outcomes[0]
        return nxt, float(reward), bool(done)

    spec = Spec(
        id=env_id,
        description=f"Gymnasium env {env_id}",
        horizon=int(horizon),
        initial_state=initial,
        step_fn=step_fn,
        actions_fn=lambda s: list(names),
        is_terminal_fn=lambda s: s in terminal,
        intended_actions=list(intended_actions),
    )
    env.close()
    return spec


def _load_module(path):
    path = os.path.abspath(path)
    name = f"rcfuzz_env_{abs(hash(path)) % 10**8}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _env_class(module):
    for value in vars(module).values():
        if isinstance(value, type) and value.__name__.endswith("Env"):
            return value
    raise AuditNotApplicable("stateful module defines no *Env class")


def stateful_spec(module_path, intended_actions=None, horizon=None,
                  max_states=200_000):
    module = _load_module(module_path)
    cls = _env_class(module)
    actions = list(module.ACTIONS)
    if intended_actions is None:
        intended_actions = list(getattr(module, "INTENDED", []))
    if horizon is None:
        horizon = int(getattr(module, "HORIZON", 40))

    probe = cls()
    probe.reset()
    initial = probe.get_state()

    def step_fn(state, action):
        env = cls()
        env.set_state(state)
        out = env.step(action)
        ns, reward, done = out[0], out[1], out[2]
        return ns, float(reward), bool(done)

    # Discover terminal states by probing done transitions.
    terminal = set()
    frontier = {initial}
    seen = set(frontier)
    for _ in range(horizon):
        nxt = set()
        for st in frontier:
            if st in terminal:
                continue
            for act in actions:
                ns, _r, done = step_fn(st, act)
                if done:
                    terminal.add(ns)
                if ns not in seen:
                    seen.add(ns)
                    nxt.add(ns)
        if len(seen) > max_states:
            raise AuditNotApplicable(
                f"state space exceeds the {max_states} cap within the "
                "horizon; narrow the horizon or the env"
            )
        frontier = nxt

    return Spec(
        id=os.path.splitext(os.path.basename(module_path))[0],
        description=f"stateful env module {module_path}",
        horizon=int(horizon),
        initial_state=initial,
        step_fn=step_fn,
        actions_fn=lambda s: list(actions),
        is_terminal_fn=lambda s: s in terminal,
        intended_actions=list(intended_actions),
    )
