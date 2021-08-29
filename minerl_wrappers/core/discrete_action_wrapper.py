import gym
import numpy as np


class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_choices=None):
        super().__init__(env)
        assert isinstance(
            self.env.action_space, gym.spaces.Box
        ), "Wrapped env must have vector action space."
        assert isinstance(action_choices, np.ndarray) or isinstance(action_choices, str)
        if isinstance(action_choices, np.ndarray):
            self.action_choices = action_choices
        elif isinstance(action_choices, str):
            self.action_choices = np.load(action_choices)
        num_actions = len(self.action_choices)
        self.action_space = gym.spaces.Discrete(num_actions)

    def action(self, action: int):
        return self.action_choices[action]

    def reverse_action(self, action: np.ndarray):
        action = np.reshape(action, (1, 64))
        distances = np.linalg.norm(action - self.action_choices, axis=1)
        return int(np.argmin(distances).item())
