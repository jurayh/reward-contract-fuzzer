#!/usr/bin/env python3
"""Build the shared head-to-head test set.

Each env is a directory in a real format ratctl understands (OpenEnv or
Gymnasium). Design/clean envs also expose get_state/set_state so the
Reward Contract Fuzzer can enumerate them as MDPs through an adapter.
The same reward constants drive both the scanned code and the MDP.

Axes (ground truth):
  tamper_vulnerable : grader/harness code can be tampered with or leaked
  design_exploitable: faithful optimisation of the reward beats intended
  clean             : neither
"""

import os
import textwrap

ROOT = os.path.join(os.path.dirname(__file__), "envs")


def write(path, content):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as fh:
        fh.write(textwrap.dedent(content))


GYM_HEADER = '''\
    """{doc}"""
    try:
        import gymnasium as gym
    except ImportError:  # minimal stub so the file imports anywhere
        class _Stub:
            class Env:
                pass
        gym = _Stub()

'''


def gym_env(name, doc, body):
    write(f"{name}/env.py",
          textwrap.dedent(GYM_HEADER.format(doc=doc)) + textwrap.dedent(body))


# ---------------------------------------------------------------- tamper ---
write("tamper_openenv/openenv.yaml", """\
    name: tamper-math-verifier
    description: OpenEnv math verifier with a tamperable grading path
    version: 1.0.0
    """)
write("tamper_openenv/server/app.py", '''\
    """OpenEnv verifier whose grading path an agent can tamper with."""
    import os
    import subprocess
    import sys
    from fastapi import FastAPI

    app = FastAPI()


    @app.post("/step")
    async def step(action: dict):
        # Reads grading tests from a writable location.
        if os.path.exists("tests/test_solution.py"):
            os.remove("tests/test_solution.py")
        # Inspects the caller to decide how strict to be.
        caller = sys._getframe(1)
        # Leaks prior solutions through repo history.
        subprocess.run(["git", "log", "-n", "1"])
        if action.get("quick_pass"):
            sys.exit(0)
        return {"observation": "passed", "reward": 1.0}
    ''')

gym_env("tamper_gym_grader", "Gymnasium env whose grader swallows failures.", '''\
    class GraderEnv(gym.Env):
        """Reward comes from a checker that returns success on any error."""

        def __init__(self):
            self.state = 0

        def reset(self, seed=None):
            self.state = 0
            return self.state, {}

        def step(self, action):
            reward = self._grade(action)
            return self.state, reward, True, False, {}

        def _grade(self, action):
            try:
                return 1.0 if action == 42 else 0.0
            except:
                return 1.0
    ''')

# ---------------------------------------------------------------- design ---
gym_env("design_farm", "Corridor cleaning env with a loose drop fine.", '''\
    ACTIONS = ["E", "W", "clean", "drop", "deposit"]
    INTENDED = ["E", "clean", "E", "deposit"]
    HORIZON = 40
    REWARD = {"clean": 10.0, "deposit": 10.0, "move": -1.0, "drop": -2.0}


    class FarmEnv(gym.Env):
        """x in 0..2; dirt at 1, bin at 2. State (x, carrying, dirt, done)."""

        def __init__(self):
            self.state = (0, 0, 1, 0)

        def reset(self, seed=None):
            self.state = (0, 0, 1, 0)
            return self.state, {}

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = tuple(state)

        def step(self, action):
            x, carrying, dirt, done = self.state
            reward = 0.0
            if action == "E":
                x = min(2, x + 1); reward = REWARD["move"]
            elif action == "W":
                x = max(0, x - 1); reward = REWARD["move"]
            elif action == "clean":
                if x == 1 and dirt == 1 and carrying == 0:
                    carrying, dirt, reward = 1, 0, REWARD["clean"]
            elif action == "drop":
                if carrying == 1:
                    carrying, reward = 0, REWARD["drop"]
                    if x == 1:
                        dirt = 1
            elif action == "deposit":
                if x == 2 and carrying == 1:
                    carrying, done, reward = 0, 1, REWARD["deposit"]
            self.state = (x, carrying, dirt, done)
            return self.state, reward, bool(done), False, {}
    ''')

gym_env("design_sensor", "Env that pays per scan at a sensor tile.", '''\
    ACTIONS = ["E", "W", "scan", "deposit"]
    INTENDED = ["E", "E", "deposit"]
    HORIZON = 40
    REWARD = {"scan": 5.0, "deposit": 10.0, "move": -1.0}


    class SensorEnv(gym.Env):
        """x in 0..2; sensor at 1, bin at 2. State (x, delivered)."""

        def __init__(self):
            self.state = (0, 0)

        def reset(self, seed=None):
            self.state = (0, 0)
            return self.state, {}

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = tuple(state)

        def step(self, action):
            x, delivered = self.state
            reward = 0.0
            if action == "E":
                x = min(2, x + 1); reward = REWARD["move"]
            elif action == "W":
                x = max(0, x - 1); reward = REWARD["move"]
            elif action == "scan":
                if x == 1:
                    reward = REWARD["scan"]
            elif action == "deposit":
                if x == 2:
                    delivered, reward = 1, REWARD["deposit"]
            self.state = (x, delivered)
            return self.state, reward, bool(delivered), False, {}
    ''')

gym_env("design_loiter", "Goal env with a per-step survival bonus.", '''\
    ACTIONS = ["E", "W", "stay"]
    INTENDED = ["E", "E", "E"]
    HORIZON = 40
    REWARD = {"survival": 1.0, "goal": 10.0}


    class LoiterEnv(gym.Env):
        """x in 0..3; goal at 3. State (x,)."""

        def __init__(self):
            self.state = (0,)

        def reset(self, seed=None):
            self.state = (0,)
            return self.state, {}

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = tuple(state)

        def step(self, action):
            (x,) = self.state
            if action == "E":
                x = min(3, x + 1)
            elif action == "W":
                x = max(0, x - 1)
            reward = REWARD["survival"]
            done = x == 3
            if done:
                reward += REWARD["goal"]
            self.state = (x,)
            return self.state, reward, done, False, {}
    ''')

gym_env("design_shaping", "Goal env with approach shaping and free retreat.", '''\
    ACTIONS = ["E", "W", "stay"]
    INTENDED = ["E", "E"]
    HORIZON = 40
    REWARD = {"closer": 2.0, "goal": 20.0}


    class ShapingEnv(gym.Env):
        """x in 0..4, start 2, goal 4. State (x,)."""

        def __init__(self):
            self.state = (2,)

        def reset(self, seed=None):
            self.state = (2,)
            return self.state, {}

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = tuple(state)

        def step(self, action):
            (x,) = self.state
            nx = x
            if action == "E":
                nx = min(4, x + 1)
            elif action == "W":
                nx = max(0, x - 1)
            reward = REWARD["closer"] if abs(4 - nx) < abs(4 - x) else 0.0
            done = nx == 4
            if done:
                reward += REWARD["goal"]
            self.state = (nx,)
            return self.state, reward, done, False, {}
    ''')

# ----------------------------------------------------------------- clean ---
gym_env("clean_delivery", "Corridor cleaning env with a heavy drop fine.", '''\
    ACTIONS = ["E", "W", "clean", "drop", "deposit"]
    INTENDED = ["E", "clean", "E", "deposit"]
    HORIZON = 40
    REWARD = {"clean": 10.0, "deposit": 10.0, "move": -1.0, "drop": -20.0}


    class CleanDeliveryEnv(gym.Env):
        """x in 0..2; dirt at 1, bin at 2. State (x, carrying, dirt, done)."""

        def __init__(self):
            self.state = (0, 0, 1, 0)

        def reset(self, seed=None):
            self.state = (0, 0, 1, 0)
            return self.state, {}

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = tuple(state)

        def step(self, action):
            x, carrying, dirt, done = self.state
            reward = 0.0
            if action == "E":
                x = min(2, x + 1); reward = REWARD["move"]
            elif action == "W":
                x = max(0, x - 1); reward = REWARD["move"]
            elif action == "clean":
                if x == 1 and dirt == 1 and carrying == 0:
                    carrying, dirt, reward = 1, 0, REWARD["clean"]
            elif action == "drop":
                if carrying == 1:
                    carrying, reward = 0, REWARD["drop"]
                    if x == 1:
                        dirt = 1
            elif action == "deposit":
                if x == 2 and carrying == 1:
                    carrying, done, reward = 0, 1, REWARD["deposit"]
            self.state = (x, carrying, dirt, done)
            return self.state, reward, bool(done), False, {}
    ''')

gym_env("clean_navigation", "Shortest-path navigation env.", '''\
    ACTIONS = ["E", "W", "stay"]
    INTENDED = ["E", "E", "E"]
    HORIZON = 20
    REWARD = {"goal": 10.0, "move": -1.0}


    class NavigationEnv(gym.Env):
        """x in 0..3, goal at 3. State (x,)."""

        def __init__(self):
            self.state = (0,)

        def reset(self, seed=None):
            self.state = (0,)
            return self.state, {}

        def get_state(self):
            return self.state

        def set_state(self, state):
            self.state = tuple(state)

        def step(self, action):
            (x,) = self.state
            reward = 0.0
            if action == "E":
                x = min(3, x + 1); reward = REWARD["move"]
            elif action == "W":
                x = max(0, x - 1); reward = REWARD["move"]
            done = x == 3
            if done:
                reward += REWARD["goal"]
            self.state = (x,)
            return self.state, reward, done, False, {}
    ''')

print("Built envs:", sorted(os.listdir(ROOT)))
