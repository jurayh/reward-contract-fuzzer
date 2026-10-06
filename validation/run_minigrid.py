#!/usr/bin/env python3
"""Broader real-env sample under the calibrated margin rule.

Envs (third-party code, installed from PyPI 2026-10-03):
  MiniGrid-Empty-5x5-v0, MiniGrid-FourRooms-v0 (minigrid on
  gymnasium 1.3.0), FrozenLake-v1 with the 8x8 map.

MiniGrid adapter: state is (x, y, direction, step_count) because
MiniGrid's goal reward is discounted by elapsed steps. The grid in
Empty and FourRooms is static (doorways, no togglable doors), so the
adapter restores agent position/direction/step count on the env's
own object and calls its real step. Audit action set is restricted
to the movement actions (left, right, forward), which is the whole
relevant policy space for these navigation tasks; pickup, drop,
toggle, and done cannot improve return here.

Intended scripts are BFS shortest paths computed against the env's
own transition behaviour, i.e. the route the task description asks
for. Scoring uses validation/calibrate.py's measure() and rules.

Run: /tmp/gym-venv/bin/python run_minigrid.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import gymnasium as gym  # noqa: E402
import minigrid  # noqa: E402,F401

from calibrate import RULES, measure  # noqa: E402
from mdp import Spec  # noqa: E402

MOVE_ACTIONS = ["left", "right", "forward"]
ACTION_IDS = {"left": 0, "right": 1, "forward": 2}


def minigrid_spec(env_id, horizon):
    env = gym.make(env_id)
    env.reset(seed=0)
    u = env.unwrapped
    goal = None
    for x in range(u.grid.width):
        for y in range(u.grid.height):
            cell = u.grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal = (x, y)
    start = (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir), 0)

    def step_fn(state, action):
        x, y, direction, steps = state
        u.agent_pos = (x, y)
        u.agent_dir = direction
        u.step_count = steps
        _obs, reward, terminated, _trunc, _info = env.step(ACTION_IDS[action])
        ns = (int(u.agent_pos[0]), int(u.agent_pos[1]),
              int(u.agent_dir), steps + 1)
        return ns, float(reward), bool(terminated)

    def bfs_script():
        prev = {start: None}
        queue = deque([start])
        found = None
        while queue:
            state = queue.popleft()
            if (state[0], state[1]) == goal:
                found = state
                break
            if state[3] >= horizon:
                continue
            for action in MOVE_ACTIONS:
                ns, _r, done = step_fn(state, action)
                if ns not in prev:
                    prev[ns] = (state, action)
                    queue.append(ns)
        assert found is not None, f"no path in {env_id}"
        script, node = [], found
        while prev[node] is not None:
            node, action = prev[node]
            script.append(action)
        return list(reversed(script))

    spec = Spec(
        id=env_id, description=f"real env {env_id}", horizon=horizon,
        initial_state=start, step_fn=step_fn,
        actions_fn=lambda s: list(MOVE_ACTIONS),
        is_terminal_fn=lambda s: (s[0], s[1]) == goal,
        intended_actions=bfs_script(), label="clean")
    env.close()
    return spec


def frozenlake8_spec():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    u = env.unwrapped
    table = u.P
    names = ["L", "D", "R", "U"]
    terminal = {s for s, pa in table.items()
                if all(all(t[3] for t in pa[a]) for a in pa)}
    holes = {s for s in terminal if s != 63}

    def step_fn(state, action):
        _p, nxt, reward, done = table[state][names.index(action)][0]
        return nxt, float(reward), bool(done)

    prev = {0: None}
    queue = deque([0])
    while queue:
        state = queue.popleft()
        if state == 63:
            break
        for a_idx, outcomes in table[state].items():
            _p, nxt, _r, _d = outcomes[0]
            if nxt not in prev and nxt not in holes:
                prev[nxt] = (state, names[a_idx])
                queue.append(nxt)
    script, node = [], 63
    while prev[node] is not None:
        node, action = prev[node]
        script.append(action)
    spec = Spec(
        id="FrozenLake-8x8", description="FrozenLake-v1 8x8 map",
        horizon=32, initial_state=0, step_fn=step_fn,
        actions_fn=lambda s: list(names),
        is_terminal_fn=lambda s: s in terminal,
        intended_actions=list(reversed(script)), label="clean")
    env.close()
    return spec


def main():
    specs = [
        minigrid_spec("MiniGrid-Empty-5x5-v0", horizon=40),
        minigrid_spec("MiniGrid-FourRooms-v0", horizon=60),
        frozenlake8_spec(),
    ]
    rows = []
    for spec in specs:
        m = measure(spec)
        m["rule_flags"] = {
            name: bool(rule(m) and m["degenerate"])
            for name, rule in RULES.items()
        }
        rows.append(m)
        flags = " ".join(f"{k.split()[0]}={'Y' if v else '.'}"
                         for k, v in m["rule_flags"].items())
        print(f"{m['id']:<24} int={m['intended']:.3f} opt={m['optimal']:.3f} "
              f"diff={m['diff']:.3f} range={m['range']:.3f} "
              f"degen={m['degenerate']}  {flags}")
    out = os.path.join(HERE, "minigrid_results.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
