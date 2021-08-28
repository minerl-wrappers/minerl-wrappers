import logging

import gym
import numpy as np
from minerl.herobraine.envs import (
    BASIC_ENV_SPECS,
    COMPETITION_ENV_SPECS,
    BASALT_COMPETITION_ENV_SPECS,
)

from minerl_wrappers import wrap

BASIC_IDS = [env_spec.name for env_spec in BASIC_ENV_SPECS]
DIAMOND_COMPETITION_IDS = [env_spec.name for env_spec in COMPETITION_ENV_SPECS]
BASALT_COMPETITION_IDS = [env_spec.name for env_spec in BASALT_COMPETITION_ENV_SPECS]

logging.basicConfig(level=logging.DEBUG)


def create_and_wrap(gym_id, config_file=None, **kwargs):
    logging.debug("Creating MineRL environment...")
    env = gym.make(gym_id)
    logging.debug("Wrapping MineRL environment...")
    env = wrap(env, config_file=config_file, **kwargs)
    return env


def reset_and_sample_episode(env, max_steps=None):
    history = []
    logging.debug("Resetting MineRL environment...")
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        if max_steps is not None and steps >= max_steps:
            break
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        history.append([obs, new_obs, reward, done, info])
        obs = new_obs
        steps += 1
        logging.debug(f"Environment loop step {steps} completed")
    return history


def build_and_run_step(gym_id, config_file=None, **kwargs):
    env = create_and_wrap(gym_id, config_file=config_file, **kwargs)
    reset_and_sample_episode(env, 1)
    logging.debug("Closing MineRL environment...")
    env.close()


def build_and_run_list_config(
    gym_id, list_config: list, max_steps=1, test_action_wrappers=False
):
    env = gym.make(gym_id)
    for config in list_config:
        logging.debug(f"testing config: {config}")
        wrapped_env = wrap(env, **config)
        wrapped_env.reset()
        if test_action_wrappers:
            assert_equal_backward = True
            if (
                config["rllib"]
                and config.get("rllib_config", {}).get("action_choices", None)
                is not None
            ):
                assert_equal_backward = False
            sample_and_test_action_wrappers(
                wrapped_env, assert_equal_backward=assert_equal_backward
            )
        for step in range(max_steps):
            action = wrapped_env.action_space.sample()
            _, _, done, _ = wrapped_env.step(action)
            if done:
                break
    env.close()


def sample_and_test_action_wrappers(
    env: gym.Wrapper, assert_equal_forward=True, assert_equal_backward=True
):
    action_wrappers = []
    env_pointer = env
    while isinstance(env_pointer, gym.Wrapper):
        if isinstance(env_pointer, gym.ActionWrapper):
            action_wrappers.append(env_pointer)
        env_pointer = env_pointer.env

    # high level action -> low level action -> high level action
    action = env.action_space.sample()

    low_level_action = action
    for wrapper in action_wrappers:
        low_level_action = wrapper.action(low_level_action)

    high_level_action = low_level_action
    for wrapper in reversed(action_wrappers):
        high_level_action = wrapper.reverse_action(high_level_action)

    if assert_equal_forward:
        np.testing.assert_equal(action, high_level_action)

    # low level action -> high level action -> low level action
    action = env.unwrapped.action_space.sample()

    high_level_action = action
    for wrapper in reversed(action_wrappers):
        high_level_action = wrapper.reverse_action(high_level_action)

    low_level_action = high_level_action
    for wrapper in action_wrappers:
        low_level_action = wrapper.action(low_level_action)

    if assert_equal_backward:
        np.testing.assert_equal(action, low_level_action)
