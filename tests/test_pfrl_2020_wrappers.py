import logging

import gym

from minerl_wrappers import wrap
from minerl_wrappers.utils import load_means
from test_wrap import reset_and_sample_episode

logging.basicConfig(level=logging.DEBUG)


def test_pfrl_2020_wrappers():
    env = gym.make("MineRLObtainDiamondDenseVectorObf-v0")
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
