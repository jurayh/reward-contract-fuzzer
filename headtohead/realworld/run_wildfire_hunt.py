#!/usr/bin/env python3
"""Hunt for a design exploit in OpenEnv's published wildfire_env.

Configuration (all published constructor knobs of the env):
  width=4, height=4   (default 32x32 is far beyond exact DP)
  humidity=1.0        (saturated air: ignition probability is exactly
                       0, so the env's own dynamics are deterministic;
                       every reward term is unchanged)
  init_sources=2, seed=3407 (the env's own default seed)
  water_capacity=8, break_capacity=50 (defaults), max_steps=1000

The step/reward code executed is the repo's own
wildfire_environment.py; only OpenEnv framework plumbing is stubbed,
as in the grid-world adapter. The MDP snapshot is
(grid, burn_timers, remaining_water, remaining_breaks, step_count).

Reward terms under test (from the env source):
  water on burning +0.25, water on fuel -0.10, break on fuel/damp
  +0.15, spread penalty -0.15/cell, shrinkage bonus +0.10/cell when
  the burning count falls, burned penalty -0.05/cell, -0.01/step,
  terminal containment bonus 0.5 + 0.5*saved_ratio + 0.2*(1-burned).

Hypotheses an optimizer can test exactly:
  H1 break farming: +0.15 for breaking any fuel cell, capacity 50,
     pays for converting the forest you were meant to save.
  H2 shrinkage farming: burning cells extinguish themselves after 3
     ticks; each burnout nets +0.10 - 0.05 = +0.05 for waiting.

Run: /tmp/gym-venv/bin/python run_wildfire_hunt.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types as pytypes

HERE = os.path.dirname(__file__)
ENV_DIR = os.path.join(HERE, "..", "vendor", "OpenEnv", "envs", "wildfire_env")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from pydantic import BaseModel  # noqa: E402


class _State(BaseModel):
    episode_id: str = ""
    step_count: int = 0


class _Obs(BaseModel):
    done: bool = False
    reward: float = 0.0


class _Env:
    def __init__(self, *a, **k):
        pass


openenv = pytypes.ModuleType("openenv")
core = pytypes.ModuleType("openenv.core")
env_server = pytypes.ModuleType("openenv.core.env_server")
interfaces = pytypes.ModuleType("openenv.core.env_server.interfaces")
env_types = pytypes.ModuleType("openenv.core.env_server.types")
env_types.State = _State
env_types.Action = BaseModel
env_types.Observation = _Obs
interfaces.Environment = _Env
interfaces.Action = BaseModel
interfaces.Observation = _Obs
interfaces.State = _State
env_server.Environment = _Env
env_server.interfaces = interfaces
env_server.types = env_types
core.env_server = env_server
openenv.core = core
sys.modules.update({
    "openenv": openenv, "openenv.core": core,
    "openenv.core.env_server": env_server,
    "openenv.core.env_server.interfaces": interfaces,
    "openenv.core.env_server.types": env_types,
})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models = load_module("wf_models", os.path.join(ENV_DIR, "models.py"))
env_path = os.path.join(ENV_DIR, "server", "wildfire_environment.py")
with open(env_path) as fh:
    source = fh.read()
source = source.replace(
    "from ..models import WildfireAction, WildfireObservation, WildfireState",
    "from wf_models import WildfireAction, WildfireObservation, WildfireState",
)
wf = pytypes.ModuleType("wf_env")
wf.__file__ = env_path
exec(compile(source, env_path, "exec"), wf.__dict__)

WildfireEnvironment = wf.WildfireEnvironment
WildfireAction = models.WildfireAction
WildfireState = models.WildfireState

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--size", type=int, default=4)
_ap.add_argument("--humidity", type=float, default=1.0)
_ap.add_argument("--rng-state", action="store_true")
_ap.add_argument("--horizon", type=int, default=30)
_ap.add_argument("--out", default="wildfire_results.json")
ARGS = _ap.parse_args()

W = H = ARGS.size
HUMIDITY = ARGS.humidity
CELLS = [(x, y) for y in range(H) for x in range(W)]
ACTIONS = (["wait"]
           + [f"water:{x}:{y}" for x, y in CELLS]
           + [f"break:{x}:{y}" for x, y in CELLS])


def make_env():
    return WildfireEnvironment(width=W, height=H, humidity=HUMIDITY,
                               init_sources=2, seed=3407,
                               water_capacity=8, break_capacity=50,
                               max_steps=1000)


def snapshot(env):
    st = env._state
    snap = (tuple(st.grid), tuple(st.burn_timers), st.remaining_water,
            st.remaining_breaks, st.step_count)
    if ARGS.rng_state:
        snap = snap + (env.rng.getstate(),)
    return snap


def restore(snap):
    grid, timers, water, breaks, steps = snap[:5]
    rng_state = snap[5] if len(snap) > 5 else None
    env = make_env()
    env._state = WildfireState.model_construct(
        episode_id="snap", step_count=steps, total_burned=0,
        total_extinguished=0, last_action="snap", width=W, height=H,
        wind_dir="CALM", humidity=HUMIDITY, remaining_water=water,
        remaining_breaks=breaks, grid=list(grid),
        burn_timers=list(timers))
    if rng_state is not None:
        env.rng.setstate(rng_state)
    else:
        env.rng.seed(12345)  # draws are < p=0.0, so rng cannot affect outcomes
    return env


def step_fn(state, action):
    env = restore(state)
    if action == "wait":
        act = WildfireAction(action="wait")
    else:
        kind, xs, ys = action.split(":")
        act = WildfireAction(action=kind, x=int(xs), y=int(ys))
    obs = env.step(act)
    return snapshot(env), float(obs.reward), bool(obs.done)


def main():
    from fuzzer import fuzz_spec
    from mdp import Spec, rollout

    probe = make_env()
    probe.reset()
    initial = snapshot(probe)
    burning = [(i % W, i // W) for i, v in enumerate(initial[0]) if v == 2]
    print("initial burning cells:", burning)

    intended = [f"water:{x}:{y}" for x, y in burning]
    spec = Spec(
        id="openenv_wildfire_4x4",
        description="OpenEnv wildfire_env 4x4, humidity=1.0, seed 3407",
        horizon=ARGS.horizon,
        initial_state=initial,
        step_fn=step_fn,
        actions_fn=lambda s: list(ACTIONS),
        is_terminal_fn=lambda s: all(v != 2 for v in s[0]),
        intended_actions=intended,
        label="unknown",
    )
    r = fuzz_spec(spec)

    wait_spec = Spec(**{**spec.__dict__, "id": "wildfire_wait_only",
                        "intended_actions": []})
    wait_traj = rollout(
        wait_spec, lambda s, t: "wait")
    intended_traj = rollout(spec, lambda s, t: (
        intended[t] if t < len(intended) else "wait"))

    result = {
        "initial_burning_cells": burning,
        "intended_script": intended,
        "intended_return": r.intended_return,
        "optimal_return": r.optimal_return,
        "diff": r.diff,
        "spike_rule": {
            "margin_threshold": r.margin_threshold,
            "flagged": r.flagged,
            "verified": r.verified,
            "terminated": r.terminated,
            "loop_ratio": r.loop_ratio,
            "top_action_share": r.top_action_share,
        },
        "wait_only_return": wait_traj.total_return,
        "optimal_actions": r.optimal_actions,
        "seconds": r.seconds,
    }
    print(json.dumps(result, indent=2))
    out = os.path.join(HERE, ARGS.out)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
