import logging

import numpy.testing

from minerl_wrappers.utils import load_means, DEFAULT_KMEANS_FILE
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
                "action_choices": DEFAULT_KMEANS_FILE,
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
        {
            "rllib": True,
            "rllib_config": {
                "normalize_action": True,
                "normalize_observation": True,
            },
        },
        {
            "rllib": True,
            "rllib_config": {
                "normalize_action": True,
                "normalize_observation": True,
                "action_choices": means,
                "include_vec_obs": False,
                "frame_skip": 4,
                "frame_stack": 4,
                "channels_first": True,
            },
        },
    ]
    list_kwargs = [{}] * len(config_list)
    discrete_action_kwargs = {"assert_equal_backward": False}
    list_kwargs[1] = discrete_action_kwargs
    list_kwargs[2] = discrete_action_kwargs
    list_kwargs[3] = discrete_action_kwargs
    list_kwargs[4] = discrete_action_kwargs
    list_kwargs[5] = discrete_action_kwargs
    list_kwargs[8] = discrete_action_kwargs

    def close(a1, a2):
        return numpy.testing.assert_almost_equal(a1, a2, decimal=6)

    list_kwargs[9] = {"forward_equality_check": close, "assert_equal_backward": False}
    list_kwargs[10] = {"assert_equal_forward": False, "assert_equal_backward": False}
    build_and_run_list_config(gym_id, config_list, list_kwargs, 4, True)
