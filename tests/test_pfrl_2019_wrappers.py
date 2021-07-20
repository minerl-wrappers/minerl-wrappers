import logging
import os
from pathlib import Path

import gym

from minerl_wrappers import wrap
from tests.common import reset_and_sample_episode

logging.basicConfig(level=logging.DEBUG)


def test_pfrl_2019_wrappers():
    env = gym.make("MineRLObtainDiamondDense-v0")
    config_validation(env)
    logging.debug("Testing default wrapper")
    wrapped_env = wrap(env)
    reset_and_sample_episode(wrapped_env, 4)
    logging.debug("Testing frame_skip and frame_stack")
    config = {
        "pfrl_2019": True,
        "pfrl_2019_config": {
            "frame_skip": 4,
            "frame_stack": 4,
        },
    }
    wrapped_env = wrap(env, **config)
    reset_and_sample_episode(wrapped_env, 4)
    logging.debug("Testing gray_scale")
    config = {
        "pfrl_2019": True,
        "pfrl_2019_config": {
            "gray_scale": True,
        },
    }
    wrapped_env = wrap(env, **config)
    reset_and_sample_episode(wrapped_env, 4)
    logging.debug("Testing random_action")
    config = {
        "pfrl_2019": True,
        "pfrl_2019_config": {
            "random_action": True,
            "eval_epsilon": 1,
        },
    }
    wrapped_env = wrap(env, **config)
    reset_and_sample_episode(wrapped_env, 4)
    env.close()


def config_validation(env):
    logging.debug("Config validation tests...")
    config = {}
    wrap(env, **config)
    config = {
        "pfrl_2019": True,
        "pfrl_2019_config": {},
    }
    wrap(env, **config)
    os.chdir(Path(__file__).absolute().parent.parent)
    config_file = str(
        Path(__file__).absolute().parent.joinpath("./configs/pfrl_2019_basic.yaml")
    )
    wrap(env, config_file=config_file)
