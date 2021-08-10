import copy

import gym
import numpy as np


class MineRLObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple(
            (self.observation_space["pov"], self.observation_space["vector"])
        )

    def observation(self, observation):
        return observation["pov"], observation["vector"]


class MineRLRemoveVecObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, pov_space_index=0):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Tuple)
        self._pov_space_index = pov_space_index
        self.observation_space = self.observation_space.spaces[self._pov_space_index]

    def observation(self, observation):
        return observation[self._pov_space_index]


class MineRLPOVChannelsLastWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space.spaces["pov"] = transpose_space(
                self.observation_space.spaces["pov"]
            )
        elif isinstance(self.observation_space, gym.spaces.Tuple):
            pov_space = transpose_space(self.observation_space.spaces[0])
            self.observation_space.spaces = (
                pov_space,
                *self.observation_space.spaces[1:],
            )
        else:
            self.observation_space = transpose_space(self.observation_space)

    def observation(self, observation):
        if isinstance(observation, dict):
            obs = copy.copy(observation)
            obs["pov"] = transpose_obs(observation["pov"])
        elif isinstance(observation, tuple):
            pov = transpose_space(observation[0])
            obs = (pov, *observation[1:])
        else:
            obs = transpose_obs(observation)
        return obs


def transpose_obs(pov_obs: np.ndarray):
    assert pov_obs.shape == [3, 64, 64]
    new_obs = pov_obs.transpose((1, 2, 0))
    assert new_obs.shape == [64, 64, 3]
    return new_obs


def transpose_space(pov_space: gym.spaces.Box):
    low = transpose_obs(pov_space.low)
    high = transpose_obs(pov_space.high)
    new_pov_space = gym.spaces.Box(low, high)
    return new_pov_space
