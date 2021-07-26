"""
July 19th, 2021
Copied from:
https://github.com/keisuke-nakata/minerl2020_submission/blob/69e837a98e446bb06e5755a9855eef5cd45bd1a8/mod/env_wrappers.py
https://github.com/keisuke-nakata/minerl2020_submission/blob/69e837a98e446bb06e5755a9855eef5cd45bd1a8/mod/dqn_family.py
TODO: update to combined introduction and research tracks
https://github.com/s-shiroshita/minerl2020_sqil_submission/blob/e5cc2c70a3b85272c8c359107fc3a5e5d53e13a1/mod/env_wrappers.py
"""
import copy
from logging import getLogger
from collections import deque
import os

import gym
import numpy as np
import cv2

from .pfrl.wrappers import ContinuingTimeLimit, RandomizeAction, Monitor
from .pfrl.wrappers.atari_wrappers import ScaledFloatFrame, LazyFrames

cv2.ocl.setUseOpenCL(False)
logger = getLogger(__name__)


def wrap_env(
    env,
    test,
    monitor,
    outdir,
    frame_skip,
    gray_scale,
    frame_stack,
    randomize_action,
    eval_epsilon,
    action_choices,
    include_vec_obs=False,
    tuple_obs_space=False,
):
    # wrap env: time limit...
    # Don't use `ContinuingTimeLimit` for testing, in order to avoid unexpected behavior on submissions.
    # (Submission utility regards "done" as an episode end, which will result in endless evaluation)
    if not test and isinstance(env, gym.wrappers.TimeLimit):
        logger.info(
            "Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit."
        )
        env = env.env
        max_episode_steps = env.spec.max_episode_steps
        env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

    # wrap env: observation...
    # NOTE: wrapping order matters!

    if test and monitor:
        env = Monitor(
            env,
            os.path.join(outdir, env.spec.id, "monitor"),
            mode="evaluation" if test else "training",
            video_callable=lambda episode_id: True,
        )
    if frame_skip is not None:
        env = FrameSkip(env, skip=frame_skip)
    if gray_scale:
        env = GrayScaleWrapper(env, dict_space_key="pov")
    if not include_vec_obs:
        env = ObtainPoVWrapper(env)
    elif tuple_obs_space:
        env = TupleObsWrapper(env)
    env = MoveAxisWrapper(
        env, source=-1, destination=0
    )  # convert hwc -> chw as Pytorch requires.
    env = ScaledFloatFrame(env)
    if frame_stack is not None and frame_stack > 0:
        env = FrameStack(env, frame_stack, channel_order="chw")

    if action_choices is not None:
        env = ClusteredActionWrapper(env, clusters=action_choices)

    if randomize_action:
        env = RandomizeAction(env, eval_epsilon)

    return env


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.

    Note that this wrapper does not "maximize" over the skipped frames.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order="hwc", use_tuple=False):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.observations = deque([], maxlen=k)
        self.stack_axis = {"hwc": 2, "chw": 0}[channel_order]
        self.use_tuple = use_tuple

        if self.use_tuple:
            pov_space = env.observation_space[0]
            inv_space = env.observation_space[1]
        else:
            pov_space = env.observation_space

        low_pov = np.repeat(pov_space.low, k, axis=self.stack_axis)
        high_pov = np.repeat(pov_space.high, k, axis=self.stack_axis)
        pov_space = gym.spaces.Box(low=low_pov, high=high_pov, dtype=pov_space.dtype)

        if self.use_tuple:
            low_inv = np.repeat(inv_space.low, k, axis=0)
            high_inv = np.repeat(inv_space.high, k, axis=0)
            inv_space = gym.spaces.Box(
                low=low_inv, high=high_inv, dtype=inv_space.dtype
            )
            self.observation_space = gym.spaces.Tuple((pov_space, inv_space))
        else:
            self.observation_space = pov_space

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.observations.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.observations.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.observations) == self.k
        if self.use_tuple:
            frames = [x[0] for x in self.observations]
            inventory = [x[1] for x in self.observations]
            return (
                LazyFrames(list(frames), stack_axis=self.stack_axis),
                LazyFrames(list(inventory), stack_axis=0),
            )
        else:
            return LazyFrames(list(self.observations), stack_axis=self.stack_axis)


class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces["pov"]

    def observation(self, observation):
        return observation["pov"]


class TupleObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple(
            (self.observation_space["pov"], self.observation_space["vector"])
        )

    def observation(self, observation):
        return observation["pov"], observation["vector"]


class UnifiedObservationWrapper(gym.ObservationWrapper):
    """Take 'pov', 'compassAngle', 'inventory' and concatenate with scaling.
    Each element of 'inventory' is converted to a square whose side length is region_size.
    The color of each square is correlated to the reciprocal of (the number of the corresponding item + 1).
    """

    def __init__(self, env, region_size=8):
        super().__init__(env)

        self._compass_angle_scale = (
            180 / 255
        )  # NOTE: `ScaledFloatFrame` will scale the pixel values with 255.0 later
        self.region_size = region_size

        pov_space = self.env.observation_space.spaces["pov"]
        low_dict = {"pov": pov_space.low}
        high_dict = {"pov": pov_space.high}

        if "compassAngle" in self.env.observation_space.spaces:
            compass_angle_space = self.env.observation_space.spaces["compassAngle"]
            low_dict["compassAngle"] = compass_angle_space.low
            high_dict["compassAngle"] = compass_angle_space.high

        if "inventory" in self.env.observation_space.spaces:
            inventory_space = self.env.observation_space.spaces["inventory"]
            low_dict["inventory"] = {}
            high_dict["inventory"] = {}
            for key in inventory_space.spaces.keys():
                low_dict["inventory"][key] = inventory_space.spaces[key].low
                high_dict["inventory"][key] = inventory_space.spaces[key].high

        low = self.observation(low_dict)
        high = self.observation(high_dict)

        self.observation_space = gym.spaces.Box(low=low, high=high)

    def observation(self, observation):
        obs = observation["pov"]
        pov_dtype = obs.dtype

        if "compassAngle" in observation:
            compass_scaled = observation["compassAngle"] / self._compass_angle_scale
            compass_channel = (
                np.ones(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype)
                * compass_scaled
            )
            obs = np.concatenate([obs, compass_channel], axis=-1)
        if "inventory" in observation:
            assert len(obs.shape[:-1]) == 2
            region_max_height = obs.shape[0]
            region_max_width = obs.shape[1]
            rs = self.region_size
            if min(region_max_height, region_max_width) < rs:
                raise ValueError("'region_size' is too large.")
            num_element_width = region_max_width // rs
            inventory_channel = np.zeros(
                shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype
            )
            for idx, key in enumerate(observation["inventory"]):
                item_scaled = np.clip(
                    255 - 255 / (observation["inventory"][key] + 1), 0, 255  # Inversed
                )
                item_channel = np.ones(shape=[rs, rs, 1], dtype=pov_dtype) * item_scaled
                width_low = (idx % num_element_width) * rs
                height_low = (idx // num_element_width) * rs
                if height_low + rs > region_max_height:
                    raise ValueError(
                        "Too many elements on 'inventory'. Please decrease 'region_size' of each component"
                    )
                inventory_channel[
                    height_low : (height_low + rs), width_low : (width_low + rs), :
                ] = item_channel
            obs = np.concatenate([obs, inventory_channel], axis=-1)
        return obs


class FullObservationSpaceWrapper(gym.ObservationWrapper):
    """Returns as observation a tuple with the frames and a list of
    compassAngle and inventory items.
    compassAngle is scaled to be in the interval [-1, 1] and inventory items
    are scaled to be in the interval [0, 1]
    """

    def __init__(self, env):
        super().__init__(env)

        pov_space = self.env.observation_space.spaces["pov"]

        low_dict = {"pov": pov_space.low, "inventory": {}}
        high_dict = {"pov": pov_space.high, "inventory": {}}

        for obs_name in self.env.observation_space.spaces["inventory"].spaces.keys():
            obs_space = self.env.observation_space.spaces["inventory"].spaces[obs_name]
            low_dict["inventory"][obs_name] = obs_space.low
            high_dict["inventory"][obs_name] = obs_space.high

        if "compassAngle" in self.env.observation_space.spaces:
            compass_angle_space = self.env.observation_space.spaces["compassAngle"]
            low_dict["compassAngle"] = compass_angle_space.low
            high_dict["compassAngle"] = compass_angle_space.high

        low = self.observation(low_dict)
        high = self.observation(high_dict)

        pov_space = gym.spaces.Box(low=low[0], high=high[0])
        inventory_space = gym.spaces.Box(low=low[1], high=high[1])
        self.observation_space = gym.spaces.Tuple((pov_space, inventory_space))

    def observation(self, observation):
        frame = observation["pov"]
        inventory = []

        if "compassAngle" in observation:
            compass_scaled = observation["compassAngle"] / 180
            inventory.append(compass_scaled)

        for obs_name in observation["inventory"].keys():
            inventory.append(observation["inventory"][obs_name] / 2304)

        inventory = np.array(inventory)
        return (frame, inventory)


class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""

    def __init__(self, env, source, destination, use_tuple=False):
        if use_tuple:
            assert isinstance(env.observation_space[0], gym.spaces.Box)
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
        super().__init__(env)

        self.source = source
        self.destination = destination
        self.use_tuple = use_tuple

        if self.use_tuple:
            low = self.observation(
                tuple([space.low for space in self.observation_space])
            )
            high = self.observation(
                tuple([space.high for space in self.observation_space])
            )
            dtype = self.observation_space[0].dtype
            pov_space = gym.spaces.Box(low=low[0], high=high[0], dtype=dtype)
            inventory_space = self.observation_space[1]
            self.observation_space = gym.spaces.Tuple((pov_space, inventory_space))
        else:
            low = self.observation(self.observation_space.low)
            high = self.observation(self.observation_space.high)
            dtype = self.observation_space.dtype
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=dtype)

    def observation(self, observation):
        if self.use_tuple:
            new_observation = list(observation)
            new_observation[0] = np.moveaxis(
                observation[0], self.source, self.destination
            )
            return tuple(new_observation)
        else:
            return np.moveaxis(observation, self.source, self.destination)


class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, dict_space_key=None):
        super().__init__(env)

        self._key = dict_space_key

        if self._key is None:
            original_space = self.observation_space
        else:
            original_space = self.observation_space.spaces[self._key]
        height, width = original_space.shape[0], original_space.shape[1]

        # sanity checks
        ideal_image_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        if original_space != ideal_image_space:
            raise ValueError(
                "Image space should be {}, but given {}.".format(
                    ideal_image_space, original_space
                )
            )
        if original_space.dtype != np.uint8:
            raise ValueError(
                "Image should `np.uint8` typed, but given {}.".format(
                    original_space.dtype
                )
            )

        height, width = original_space.shape[0], original_space.shape[1]
        new_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )
        if self._key is None:
            self.observation_space = new_space
        else:
            new_space_dict = copy.deepcopy(self.observation_space)
            new_space_dict.spaces[self._key] = new_space
            self.observation_space = new_space_dict

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        if self._key is None:
            obs = frame
        else:
            obs[self._key] = frame
        return obs


class ClusteredActionWrapper(gym.ActionWrapper):
    def __init__(self, env, clusters):
        super().__init__(env)
        self._clusters = clusters

        self._np_random = np.random.RandomState()

        self.action_space = gym.spaces.Discrete(len(clusters))

    def action(self, action):
        return {"vector": self._clusters[action]}

    def seed(self, seed):
        super().seed(seed)
        self._np_random.seed(seed)
