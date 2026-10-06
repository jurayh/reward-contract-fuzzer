#!/usr/bin/env python3
"""Run the Reward Contract Fuzzer on the head-to-head envs.

Each Gymnasium-style env exposes get_state/set_state, so the adapter
enumerates it as an MDP and reuses the spike's DP optimizer and
replay-verification fuzzer unchanged. Tamper envs have no MDP adapter:
their exploit lives in grader code, outside the reward MDP, and the
harness records them as out-of-model rather than pretending to score
them.

Usage: python3 run_fuzzer.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fuzzer import fuzz_spec  # noqa: E402
from mdp import Spec  # noqa: E402

ENVS = [
    "design_farm",
    "design_sensor",
    "design_loiter",
    "design_shaping",
    "clean_delivery",
    "clean_navigation",
]


def load_env(name):
    path = os.path.join(HERE, "envs", name, "env.py")
    module_spec = importlib.util.spec_from_file_location(f"hh_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def env_class(module):
    for value in vars(module).values():
        if isinstance(value, type) and value.__name__.endswith("Env"):
            return value
    raise RuntimeError("no Env class found")


def to_spec(name, module) -> Spec:
    cls = env_class(module)
    probe = cls()
    initial, _ = probe.reset()

    def step_fn(state, action):
        env = cls()
        env.set_state(state)
        ns, reward, done, _trunc, _info = env.step(action)
        return ns, float(reward), bool(done)

    def is_terminal(state):
        # Terminal is encoded by the step that produced the state; for
        # enumeration we treat a state as terminal when every action
        # leaves the done flag set. Simpler: re-derive via a marker.
        return getattr(to_spec, "_terminal", {}).get(state, False)

    spec = Spec(
        id=name,
        description=f"head-to-head env {name}",
        horizon=int(module.HORIZON),
        initial_state=initial,
        step_fn=step_fn,
        actions_fn=lambda s: list(module.ACTIONS),
        is_terminal_fn=is_terminal,
        intended_actions=list(module.INTENDED),
        label="unknown",
    )
    # Precompute terminal states by probing: a state is terminal if the
    # env reports done when *entering* it. Our envs encode done only on
    # the transition, so wrap step to record terminal successors.
    terminal = set()

    def step_recording(state, action):
        ns, reward, done = step_fn(state, action)
        if done:
            terminal.add(ns)
        return ns, reward, done

    spec.step_fn = step_recording
    # Discover terminal states with a breadth pass, then rebuild cleanly.
    frontier = {initial}
    seen = set(frontier)
    for _ in range(spec.horizon):
        nxt = set()
        for st in frontier:
            if st in terminal:
                continue
            for act in module.ACTIONS:
                ns, _r, done = step_fn(st, act)
                if done:
                    terminal.add(ns)
                if ns not in seen:
                    seen.add(ns)
                    nxt.add(ns)
        frontier = nxt
    to_spec._terminal = terminal  # type: ignore[attr-defined]
    spec.is_terminal_fn = lambda s: s in terminal
    return spec


def main():
    results = {}
    for name in ENVS:
        module = load_env(name)
        spec = to_spec(name, module)
        result = fuzz_spec(spec)
        results[name] = {
            "intended_return": result.intended_return,
            "optimal_return": result.optimal_return,
            "diff": result.diff,
            "margin_threshold": result.margin_threshold,
            "flagged": result.flagged,
            "verified": result.verified,
            "terminated": result.terminated,
            "loop_ratio": result.loop_ratio,
            "top_action_share": result.top_action_share,
            "seconds": result.seconds,
        }
        print(f"{name:<18} intended={result.intended_return:>7.1f} "
              f"optimal={result.optimal_return:>7.1f} diff={result.diff:>7.1f} "
              f"flagged={result.flagged} verified={result.verified}")
    with open(os.path.join(HERE, "fuzzer_out.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("Wrote fuzzer_out.json")


if __name__ == "__main__":
    main()
