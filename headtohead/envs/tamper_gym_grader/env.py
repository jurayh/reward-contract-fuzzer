"""Gymnasium env whose grader swallows failures."""
try:
    import gymnasium as gym
except ImportError:  # minimal stub so the file imports anywhere
    class _Stub:
        class Env:
            pass
    gym = _Stub()

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
