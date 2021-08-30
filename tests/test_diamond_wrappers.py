import logging

import numpy.testing

from minerl_wrappers.utils import load_means, DEFAULT_KMEANS_FILE
from tests.common import build_and_run_list_config

logging.basicConfig(level=logging.DEBUG)


def test_diamond_wrappers():
    gym_id = "MineRLObtainDiamondDenseVectorObf-v0"
    means = load_means()
    config_list = [
        {
            "diamond": True,
        },
        {
            "diamond": True,
            "diamond_config": {
                "action_choices": means,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "action_choices": "debug",
                "frame_skip": 4,
                "frame_stack": 4,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "action_choices": DEFAULT_KMEANS_FILE,
                "gray_scale": True,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "action_choices": means,
                "seed": 0,
                "reward_scale": 2.0,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "action_choices": means,
                "include_vec_obs": False,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "include_vec_obs": False,
                "frame_skip": 4,
                "frame_stack": 4,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "channels_first": True,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "action_choices": means,
                "include_vec_obs": False,
                "frame_skip": 4,
                "frame_stack": 4,
                "channels_first": True,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "normalize_action": True,
                "normalize_observation": True,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "normalize_action": True,
                "normalize_observation": True,
                "action_choices": means,
                "include_vec_obs": False,
                "frame_skip": 4,
                "frame_stack": 4,
                "channels_first": True,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "tuple_obs_space": False,
                "flatten_action_space": False,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "tuple_obs_space": False,
                "flatten_action_space": False,
                "channels_first": True,
            },
        },
        {
            "diamond": True,
            "diamond_config": {
                "tuple_obs_space": False,
                "flatten_action_space": False,
                "normalize_action": True,
            },
        },
    ]
    list_kwargs = [{}] * len(config_list)
    discrete_action = {"assert_equal_backward": False}
    no_reset = {"needs_reset": False, "needs_time_limit": False}
    discrete_action_no_reset = {"assert_equal_backward": False}
    discrete_action_no_reset.update(no_reset)
    list_kwargs[1] = discrete_action_no_reset
    list_kwargs[2] = discrete_action
    list_kwargs[3] = discrete_action_no_reset
    list_kwargs[4] = discrete_action
    list_kwargs[5] = discrete_action_no_reset
    list_kwargs[7] = no_reset
    list_kwargs[8] = discrete_action

    def close(a1, a2):
        return numpy.testing.assert_almost_equal(a1, a2, decimal=6)

    list_kwargs[9] = {"forward_equality_check": close, "assert_equal_backward": False}
    list_kwargs[10] = {"assert_equal_forward": False, "assert_equal_backward": False}
    list_kwargs[11] = no_reset
    list_kwargs[12] = no_reset
    list_kwargs[13] = {"assert_equal_forward": False, "assert_equal_backward": False}
    build_and_run_list_config(gym_id, config_list, list_kwargs, 4, True)
