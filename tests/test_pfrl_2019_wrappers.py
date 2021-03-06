import logging
import os
from pathlib import Path

import gym

from minerl_wrappers import wrap
from tests.common import build_and_run_list_config

logging.basicConfig(level=logging.DEBUG)


def test_pfrl_2019_wrappers():
    gym_id = "MineRLObtainDiamondDense-v0"
    config_validation(gym_id)
    config_list = [
        {
            "pfrl_2019": True,
        },
        {
            "pfrl_2019": True,
            "pfrl_2019_config": {
                "frame_skip": 4,
                "frame_stack": 4,
            },
        },
        {
            "pfrl_2019": True,
            "pfrl_2019_config": {
                "gray_scale": True,
            },
        },
        {
            "pfrl_2019": True,
            "pfrl_2019_config": {
                "random_action": True,
                "eval_epsilon": 1,
            },
        },
    ]
    build_and_run_list_config(gym_id, config_list, max_steps=4)


def config_validation(gym_id):
    logging.debug("Config validation tests...")
    env = gym.make(gym_id)
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
    env.close()
