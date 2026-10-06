#!/usr/bin/env python3
"""Run the Reward Contract Fuzzer on OpenEnv's real grid_world_env.

The environment under test is Meta's OpenEnv example env
(envs/grid_world_env, cloned from github.com/meta-pytorch/OpenEnv):
5x5 grid, start (0,0), goal (4,4), reward -0.1 per step and +1.0 at
the goal. The step/reward logic executed here is the repo's own
grid_world_environment.py. Only the framework plumbing (OpenEnv base
classes, pydantic action/observation types) is stubbed, because the
full server stack (fastapi etc.) is irrelevant to reward semantics;
the import line for the models module is redirected to the repo's own
models.py loaded from disk. No transition or reward code is rewritten.

Run: /tmp/gym-venv/bin/python run_openenv_fuzzer.py  (needs pydantic)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types as pytypes

HERE = os.path.dirname(__file__)
ENV_DIR = os.path.join(HERE, "..", "vendor", "OpenEnv", "envs", "grid_world_env")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

# --- Stub the OpenEnv framework plumbing (not the env logic) -------------
from pydantic import BaseModel, Field  # noqa: E402


class _State(BaseModel):
    episode_id: str = ""
    step_count: int = 0


class _Env:
    def __init__(self, *a, **k):
        pass


openenv = pytypes.ModuleType("openenv")
core = pytypes.ModuleType("openenv.core")
env_server = pytypes.ModuleType("openenv.core.env_server")
env_types = pytypes.ModuleType("openenv.core.env_server.types")
env_types.State = _State
env_types.Action = BaseModel
env_types.Observation = BaseModel
env_server.Environment = _Env
env_server.types = env_types
core.env_server = env_server
openenv.core = core
sys.modules.update({
    "openenv": openenv,
    "openenv.core": core,
    "openenv.core.env_server": env_server,
    "openenv.core.env_server.types": env_types,
})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models = load_module("gw_models", os.path.join(ENV_DIR, "models.py"))

# Load the repo's env file; redirect only its relative models import.
env_path = os.path.join(ENV_DIR, "server", "grid_world_environment.py")
with open(env_path) as fh:
    source = fh.read()
source = source.replace(
    "from ..models import GridWorldAction, GridWorldObservation, MoveAction",
    "from gw_models import GridWorldAction, GridWorldObservation, MoveAction",
)
gw = pytypes.ModuleType("gw_env")
gw.__file__ = env_path
exec(compile(source, env_path, "exec"), gw.__dict__)

GridWorldEnvironment = gw.GridWorldEnvironment
MoveAction = models.MoveAction
GridWorldAction = models.GridWorldAction

# --- MDP adapter ----------------------------------------------------------
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
GOAL = (4, 4)


def step_fn(state, action):
    env = GridWorldEnvironment()
    env.agent_x, env.agent_y = state
    obs = env.step(GridWorldAction(action=MoveAction(action)))
    return (obs.x, obs.y), float(obs.reward), bool(obs.done)


def main():
    from fuzzer import fuzz_spec
    from mdp import Spec

    spec = Spec(
        id="openenv_grid_world",
        description="Meta OpenEnv grid_world_env (5x5, -0.1/step, +1 goal)",
        horizon=20,
        initial_state=(0, 0),
        step_fn=step_fn,
        actions_fn=lambda s: list(ACTIONS),
        is_terminal_fn=lambda s: s == GOAL,
        intended_actions=["DOWN"] * 4 + ["RIGHT"] * 4,
        label="unknown",
    )
    r = fuzz_spec(spec)
    result = {
        "intended_return": r.intended_return,
        "optimal_return": r.optimal_return,
        "diff": r.diff,
        "margin_threshold": r.margin_threshold,
        "flagged": r.flagged,
        "verified": r.verified,
        "terminated": r.terminated,
        "seconds": r.seconds,
    }
    print(f"openenv_grid_world intended={r.intended_return:.2f} "
          f"optimal={r.optimal_return:.2f} diff={r.diff:.2f} "
          f"flagged={r.flagged} verified={r.verified}")
    out = os.path.join(HERE, "openenv_results.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
