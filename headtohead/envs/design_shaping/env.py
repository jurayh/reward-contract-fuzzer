"""Goal env with approach shaping and free retreat."""
try:
    import gymnasium as gym
except ImportError:  # minimal stub so the file imports anywhere
    class _Stub:
        class Env:
            pass
    gym = _Stub()

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
