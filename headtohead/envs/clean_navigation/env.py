"""Shortest-path navigation env."""
try:
    import gymnasium as gym
except ImportError:  # minimal stub so the file imports anywhere
    class _Stub:
        class Env:
            pass
    gym = _Stub()

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
