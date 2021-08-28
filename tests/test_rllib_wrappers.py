import logging

from minerl_wrappers.utils import load_means
from tests.common import build_and_run_list_config

logging.basicConfig(level=logging.DEBUG)


def test_rllib_wrappers():
    gym_id = "MineRLObtainDiamondDenseVectorObf-v0"
    means = load_means()
    config_list = [
        {
            "rllib": True,
        },
        {
            "rllib": True,
            "rllib_config": {
                "action_choices": means,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "action_choices": means,
                "frame_skip": 4,
                "frame_stack": 4,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "action_choices": means,
                "gray_scale": True,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "action_choices": means,
                "seed": 0,
                "reward_scale": 2.0,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "action_choices": means,
                "include_vec_obs": False,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "include_vec_obs": False,
                "frame_skip": 4,
                "frame_stack": 4,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "channels_first": True,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "action_choices": means,
                "include_vec_obs": False,
                "frame_skip": 4,
                "frame_stack": 4,
                "channels_first": True,
            },
        },
    ]
    build_and_run_list_config(gym_id, config_list, 4, True)
