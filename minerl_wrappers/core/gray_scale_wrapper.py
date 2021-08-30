import gym
import numpy as np

from .observation_wrapper import MineRLPOVTransformationWrapper


class MineRLGrayScale(MineRLPOVTransformationWrapper):
    def transform_pov_space(self, pov_space):
        low = np.min(pov_space.low, axis=2, keepdims=True)
        high = np.max(pov_space.high, axis=2, keepdims=True)
        return gym.spaces.Box(low, high, dtype=pov_space.dtype)

    def transform_pov(self, pov):
        return np.mean(pov, axis=2, keepdims=True)
