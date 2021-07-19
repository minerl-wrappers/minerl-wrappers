import logging
import os
from pathlib import Path

import gym

from minerl_wrappers import wrap
from minerl_wrappers.utils import load_means
from test_diamond_competition import reset_and_sample_episode

logging.basicConfig(level=logging.DEBUG)


def test_pfrl_2020_wrappers():
    env = gym.make("MineRLObtainDiamondDenseVectorObf-v0")
    config_validation(env)
    means = load_means()
    logging.debug("Testing default wrapper")
    wrapped_env = wrap(env)
    reset_and_sample_episode(wrapped_env, 4)
    logging.debug("Testing preset action choices")
    config = {
        "pfrl_2020": True,
        "pfrl_2020_config": {
            "action_choices": means,
        },
    }
    wrapped_env = wrap(env, **config)
    reset_and_sample_episode(wrapped_env, 4)
    logging.debug("Testing frame_skip and frame_stack")
    config = {
        "pfrl_2020": True,
        "pfrl_2020_config": {
            "action_choices": means,
            "frame_skip": 4,
            "frame_stack": 4,
        },
    }
    wrapped_env = wrap(env, **config)
    reset_and_sample_episode(wrapped_env, 4)
    logging.debug("Testing gray_scale")
    config = {
        "pfrl_2020": True,
        "pfrl_2020_config": {
            "action_choices": means,
            "gray_scale": True,
        },
    }
    wrapped_env = wrap(env, **config)
    reset_and_sample_episode(wrapped_env, 4)
    logging.debug("Testing random_action")
    config = {
        "pfrl_2020": True,
        "pfrl_2020_config": {
            "action_choices": means,
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
        "pfrl_2020": True,
        "pfrl_2020_config": {
            "action_choices": None,
        },
    }
    wrap(env, **config)
    config["pfrl_2020_config"]["action_choices"] = 2
    try:
        wrap(env, **config)
    except ValueError:
        logging.debug("Intentionally caught error")
    config["pfrl_2020_config"]["action_choices"] = load_means()
    wrap(env, **config)
    os.chdir(Path(__file__).absolute().parent.parent)
    config_file = str(
        Path(__file__).absolute().parent.joinpath("./configs/pfrl_2020_basic.yaml")
    )
    wrap(env, config_file=config_file)
