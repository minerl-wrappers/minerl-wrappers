import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import TransformReward

from .action_wrapper import MineRLActionTransformationWrapper
from .observation_wrapper import MineRLObservationTransformationWrapper


def normalize(a, prev_low, prev_high, new_low, new_high):
    return (a - prev_low) / (prev_high - prev_low) * (new_high - new_low) + new_low


class MineRLNormalizeObservationWrapper(MineRLObservationTransformationWrapper):
    def __init__(self, env, pov_low=0.0, pov_high=1.0, vec_low=-1.0, vec_high=1.0):
        self.pov_low = pov_low
        self.pov_high = pov_high
        self.vec_low = vec_low
        self.vec_high = vec_high
        super().__init__(env)

    def transform_pov_space(self, pov_space):
        return gym.spaces.Box(
            self.pov_low, self.pov_high, self.old_pov_space.low.shape, np.float32
        )

    def transform_vec_space(self, vec_space):
        return gym.spaces.Box(
            self.vec_low, self.vec_high, self.old_vec_space.low.shape, np.float32
        )

    def transform_pov(self, pov):
        return normalize(
            pov,
            self.old_pov_space.low,
            self.old_pov_space.high,
            self.pov_space.low,
            self.pov_space.high,
        )

    def transform_vec(self, vec):
        return normalize(
            vec,
            self.old_vec_space.low,
            self.old_vec_space.high,
            self.vec_space.low,
            self.vec_space.high,
        )


class MineRLNormalizeActionWrapper(MineRLActionTransformationWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        self.low = low
        self.high = high
        super().__init__(env)

    def transform_vec_space(self, vec_space: Box) -> Box:
        return Box(self.low, self.high, self.old_vec_space.low.shape, np.float32)

    def transform_vec(self, vec):
        return normalize(
            vec,
            self.old_vec_space.low,
            self.old_vec_space.high,
            self.vec_space.low,
            self.vec_space.high,
        )

    def reverse_transform_vec(self, vec):
        return normalize(
            vec,
            self.vec_space.low,
            self.vec_space.high,
            self.old_vec_space.low,
            self.old_vec_space.high,
        )


class MineRLRewardScaleWrapper(TransformReward):
    def __init__(self, env, reward_scale=1.0):
        def f(reward):
            return reward * reward_scale

        super().__init__(env, f)
