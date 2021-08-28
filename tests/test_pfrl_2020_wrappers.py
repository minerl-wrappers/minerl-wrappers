import logging
import os
from pathlib import Path

import gym

from minerl_wrappers import wrap
from minerl_wrappers.utils import load_means
from tests.common import build_and_run_list_config

logging.basicConfig(level=logging.DEBUG)


def test_pfrl_2020_wrappers():
    gym_id = "MineRLObtainDiamondDenseVectorObf-v0"
    config_validation(gym_id)
    means = load_means()
    config_list = [
        {
            "pfrl_2020": True,
        },
        {
            "pfrl_2020": True,
            "pfrl_2020_config": {
                "action_choices": means,
            },
        },
        {
            "pfrl_2020": True,
            "pfrl_2020_config": {
                "action_choices": means,
                "frame_skip": 4,
                "frame_stack": 4,
            },
        },
        {
            "pfrl_2020": True,
            "pfrl_2020_config": {
                "action_choices": means,
                "gray_scale": True,
            },
        },
        {
            "pfrl_2020": True,
            "pfrl_2020_config": {
                "action_choices": means,
                "random_action": True,
                "eval_epsilon": 1,
            },
        },
        {
            "pfrl_2020": True,
            "pfrl_2020_config": {
                "action_choices": means,
                "include_vec_obs": True,
            },
        },
        {
            "pfrl_2020": True,
            "pfrl_2020_config": {
                "action_choices": means,
                "include_vec_obs": True,
                "tuple_obs_space": True,
            },
        },
    ]
    build_and_run_list_config(gym_id, config_list, 4)


def config_validation(gym_id):
    logging.debug("Config validation tests...")
    env = gym.make(gym_id)
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
    env.close()
