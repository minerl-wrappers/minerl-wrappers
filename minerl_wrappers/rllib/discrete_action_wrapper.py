import gym
import numpy as np
from sklearn.neighbors import NearestNeighbors


class MineRLDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, action_choices=None):
        super().__init__(env)
        if isinstance(action_choices, np.ndarray):
            self.action_choices = action_choices
        elif isinstance(action_choices, str):
            self.action_choices = np.load(action_choices)
        else:
            raise ValueError("must set file_path or action_choices")
        num_actions = len(self.action_choices)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(
            self.action_choices
        )

    def action(self, action: int):
        return self.action_choices[action]

    def reverse_action(self, action: np.ndarray):
        action = np.reshape(action, (1, 64))
        distances, indices = self.nearest_neighbors.kneighbors(action)
        return int(indices[0].item())
