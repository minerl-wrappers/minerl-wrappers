import gym
import numpy as np
from gym.wrappers import TransformReward

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


class MineRLNormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self._old_vec_space: gym.spaces.Box = self.action_space
        assert isinstance(self._old_vec_space, gym.spaces.Box)
        self._vec_space = gym.spaces.Box(
            low, high, self._old_vec_space.low.shape, np.float32
        )
        self.action_space = self._vec_space

    def action(self, action):
        return normalize(
            action,
            self._old_vec_space.low,
            self._old_vec_space.high,
            self._vec_space.low,
            self._vec_space.high,
        )

    def reverse_action(self, action):
        return normalize(
            action,
            self._vec_space.low,
            self._vec_space.high,
            self._old_vec_space.low,
            self._old_vec_space.high,
        )


class MineRLRewardScaleWrapper(TransformReward):
    def __init__(self, env, reward_scale=1.0):
        def f(reward):
            return reward * reward_scale

        super().__init__(env, f)
