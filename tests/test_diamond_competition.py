import logging
import time

import gym
import pytest
from minerl.herobraine.envs import (
    BASIC_ENV_SPECS,
    COMPETITION_ENV_SPECS,
    BASALT_COMPETITION_ENV_SPECS,
)

from minerl_wrappers import wrap
from minerl_wrappers.utils import load_means

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
    time.sleep(10)  # let processes clean up


@pytest.mark.parametrize("gym_id", DIAMOND_COMPETITION_IDS)
def test_diamond_competition_envs(gym_id):
    logging.debug("Loading kmeans")
    means = load_means()
    build_and_run_step(
        gym_id, pfrl_2020=True, pfrl_2020_config={"action_choices": means}
    )
    logging.debug("Finished test!")