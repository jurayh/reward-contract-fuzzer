#!/usr/bin/env python3
"""Track 2b: sampled search on a genuinely model-free stochastic env.

OpenEnv wildfire_env at its default humidity (0.25): fire spread is
stochastic, no transition table exists, and the grid is far too large
for exact DP at published size. Configured to 4x4 through the env's
own constructor knobs so episodes are cheap; every reward term and
the spread model are the published ones.

There is no policy table to search (the state is a whole grid), so
the search space is a family of reactive heuristic policies:

  burning cells exist -> water one, chosen by:
      first    : first burning cell in row order (the intended rule)
      cluster  : burning cell with most burning neighbours
      risk     : burning cell with most fuel neighbours
  no burning cells -> idle action:
      wait     : do nothing (the intended rule)
      ring     : firebreak a fuel cell next to ash
      prewet   : water the fuel cell with most fuel neighbours
      wall     : firebreak the fuel cell with most fuel neighbours

12 family members + wait-only. Each is scored on the same 300
episode seeds (paired comparison); the env is re-seeded per episode,
so scores are means over fire placements, wind, and spread streams.

Question: does any family member beat the intended rule (first/wait)
by the calibrated margin in mean return, beyond noise? If a simple
search finds a materially better *honest* policy, that is baseline
weakness again, not an exploit, and the spike should say so.

Run: /tmp/gym-venv/bin/python track2b_wildfire.py  (needs pydantic)
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types as pytypes

HERE = os.path.dirname(__file__)
ENV_DIR = os.path.join(HERE, "..", "headtohead", "vendor", "OpenEnv",
                       "envs", "wildfire_env")

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


models = load_module("wf2_models", os.path.join(ENV_DIR, "models.py"))
env_path = os.path.join(ENV_DIR, "server", "wildfire_environment.py")
with open(env_path) as fh:
    source = fh.read()
source = source.replace(
    "from ..models import WildfireAction, WildfireObservation, WildfireState",
    "from wf2_models import WildfireAction, WildfireObservation, "
    "WildfireState")
wf = pytypes.ModuleType("wf2_env")
wf.__file__ = env_path
exec(compile(source, env_path, "exec"), wf.__dict__)

WildfireEnvironment = wf.WildfireEnvironment
WildfireAction = models.WildfireAction

W = H = 4
EPISODES = 300
MAX_STEPS = 60


def neighbours(x, y):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                yield nx, ny


def choose_action(grid, pick, idle):
    def cell(x, y):
        return grid[y * W + x]

    burning = [(x, y) for y in range(H) for x in range(W) if cell(x, y) == 2]
    if burning:
        if pick == "first":
            x, y = burning[0]
        elif pick == "cluster":
            x, y = max(burning, key=lambda c: sum(
                1 for n in neighbours(*c) if cell(*n) == 2))
        else:  # risk
            x, y = max(burning, key=lambda c: sum(
                1 for n in neighbours(*c) if cell(*n) == 1))
        return WildfireAction(action="water", x=x, y=y)
    if idle == "wait":
        return WildfireAction(action="wait")
    fuel = [(x, y) for y in range(H) for x in range(W) if cell(x, y) == 1]
    if not fuel:
        return WildfireAction(action="wait")
    if idle == "ring":
        ring = [c for c in fuel
                if any(cell(*n) == 0 for n in neighbours(*c))]
        if ring:
            return WildfireAction(action="break", x=ring[0][0],
                                  y=ring[0][1])
        return WildfireAction(action="wait")
    target = max(fuel, key=lambda c: sum(
        1 for n in neighbours(*c) if cell(*n) == 1))
    kind = "water" if idle == "prewet" else "break"
    return WildfireAction(action=kind, x=target[0], y=target[1])


def run_episode(pick, idle, seed):
    env = WildfireEnvironment(width=W, height=H, humidity=0.25,
                              init_sources=2, seed=seed,
                              water_capacity=8, break_capacity=50,
                              max_steps=MAX_STEPS)
    env.reset()
    total = 0.0
    for _ in range(MAX_STEPS):
        grid = list(env._state.grid)
        obs = env.step(choose_action(grid, pick, idle))
        total += float(obs.reward)
        if obs.done:
            break
    return total


def evaluate(pick, idle, seeds):
    returns = [run_episode(pick, idle, s) for s in seeds]
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    se = math.sqrt(var / len(returns))
    return mean, se


def main():
    seeds = [50_000 + i for i in range(EPISODES)]
    family = [(p, i) for p in ("first", "cluster", "risk")
              for i in ("wait", "ring", "prewet", "wall")]
    results = {}
    for pick, idle in family:
        mean, se = evaluate(pick, idle, seeds)
        results[f"{pick}/{idle}"] = {"mean": mean, "se": se}
        print(f"{pick:>7}/{idle:<6} mean {mean:7.3f}  se {se:.3f}")
    mean, se = evaluate("first", "wait", seeds)  # intended, already in
    wait_mean, wait_se = None, None
    # wait-only baseline
    def wait_only(seed):
        env = WildfireEnvironment(width=W, height=H, humidity=0.25,
                                  init_sources=2, seed=seed,
                                  water_capacity=8, break_capacity=50,
                                  max_steps=MAX_STEPS)
        env.reset()
        total = 0.0
        for _ in range(MAX_STEPS):
            obs = env.step(WildfireAction(action="wait"))
            total += float(obs.reward)
            if obs.done:
                break
        return total
    returns = [wait_only(s) for s in seeds]
    wait_mean = sum(returns) / len(returns)
    wait_se = math.sqrt(sum((r - wait_mean) ** 2 for r in returns)
                        / len(returns) / len(returns))
    results["wait-only"] = {"mean": wait_mean, "se": wait_se}
    print(f"{'wait-only':>13} mean {wait_mean:7.3f}  se {wait_se:.3f}")

    intended = results["first/wait"]["mean"]
    best_name = max(results, key=lambda k: results[k]["mean"])
    best = results[best_name]["mean"]
    gap = best - intended
    print(f"\nintended (first/wait): {intended:.3f}")
    print(f"best found: {best_name} at {best:.3f} (gap {gap:+.3f})")
    out = os.path.join(HERE, "track2b_results.json")
    with open(out, "w") as fh:
        json.dump({"results": results, "intended": intended,
                   "best": best_name, "gap": gap,
                   "episodes": EPISODES}, fh, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
