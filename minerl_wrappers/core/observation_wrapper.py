from abc import ABC

import gym
import numpy as np
from gym.spaces import Box


class MineRLObservationTransformationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.old_pov_space = self.env.observation_space.spaces["pov"]
            self.old_vec_space = self.env.observation_space.spaces["vector"]
            self.pov_space = self.transform_pov_space(self.old_pov_space)
            self.vec_space = self.transform_vec_space(self.old_vec_space)
            spaces = {"pov": self.pov_space, "vec": self.vec_space}
            self.observation_space = gym.spaces.Dict(spaces)
        elif isinstance(self.env.observation_space, gym.spaces.Tuple):
            self.old_pov_space = self.env.observation_space.spaces[0]
            self.old_vec_space = self.env.observation_space.spaces[1]
            self.pov_space = self.transform_pov_space(self.old_pov_space)
            self.vec_space = self.transform_vec_space(self.old_vec_space)
            spaces = (self.pov_space, self.vec_space)
            self.observation_space = gym.spaces.Tuple(spaces)
        else:
            self.old_pov_space = self.env.observation_space
            self.pov_space = self.transform_pov_space(self.old_pov_space)
            assert isinstance(self.pov_space, gym.Space)
            self.observation_space = self.pov_space

    def observation(self, observation):
        if isinstance(observation, dict):
            pov = self.transform_pov(observation["pov"])
            vec = self.transform_vec(observation["vector"])
            return {"pov": pov, "vector": vec}
        elif isinstance(observation, tuple):
            pov = self.transform_pov(observation[0])
            vec = self.transform_vec(observation[1])
            return pov, vec
        else:
            return self.transform_pov(observation)

    def transform_pov_space(self, pov_space: Box) -> Box:
        raise NotImplementedError

    def transform_vec_space(self, vec_space: Box) -> Box:
        raise NotImplementedError

    def transform_pov(self, pov):
        raise NotImplementedError

    def transform_vec(self, vec):
        raise NotImplementedError


class MineRLPOVTransformationWrapper(MineRLObservationTransformationWrapper, ABC):
    def transform_vec_space(self, vec_space):
        return vec_space

    def transform_vec(self, vec):
        return vec


class MineRLTupleObservationWrapper(gym.ObservationWrapper):
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


class MineRLPOVChannelsFirstWrapper(MineRLPOVTransformationWrapper):
    def transform_pov_space(self, pov_space):
        return transpose_space(pov_space)

    def transform_pov(self, pov):
        return transpose_obs(pov)


def transpose_obs(pov_obs: np.ndarray):
    assert pov_obs.shape == (64, 64, 3)
    new_obs = pov_obs.transpose((2, 0, 1))
    assert new_obs.shape == (3, 64, 64)
    return new_obs


def transpose_space(pov_space: gym.spaces.Box):
    low = transpose_obs(pov_space.low)
    high = transpose_obs(pov_space.high)
    new_pov_space = gym.spaces.Box(low, high)
    return new_pov_space
