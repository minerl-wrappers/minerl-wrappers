import os
from typing import Callable, Any

import numpy as np
import yaml

from .pfrl_2020_wrappers import wrap_env as pfrl_2020_wrap_env
from .pfrl_2019_wrappers import wrap_env as pfrl_2019_wrap_env
from .diamond_wrappers import wrap_env as diamond_wrap_env
from .utils import merge_dicts, load_means, get_env_id

DEFAULT_CONFIG = {
    "pfrl_2019": False,
    "pfrl_2019_config": {
        "test": False,
        "monitor": False,
        "outdir": "results",
        "frame_skip": None,
        "gray_scale": False,
        "frame_stack": None,
        "disable_action_prior": False,
        "always_keys": None,
        "reverse_keys": None,
        "exclude_keys": None,
        "exclude_noop": False,
        "randomize_action": False,
        "eval_epsilon": 0.001,
    },
    "pfrl_2020": False,
    "pfrl_2020_config": {
        "test": False,
        "monitor": False,
        "outdir": "results",
        "frame_skip": None,
        "gray_scale": False,
        "frame_stack": None,
        "randomize_action": False,
        "eval_epsilon": 0.001,
        "action_choices": None,
        "include_vec_obs": False,
        "tuple_obs_space": False,
    },
    "diamond": False,
    "diamond_config": {
        "frame_stack": 1,
        "frame_skip": 1,
        "gray_scale": False,
        "seed": None,
        "normalize_observation": False,
        "normalize_action": False,
        "reward_scale": 1.0,
        "action_choices": None,
        "include_vec_obs": True,
        "channels_first": False,
        "tuple_obs_space": True,
        "flatten_action_space": True,
    },
}

CONFLICTING_OPTIONS = ["pfrl_2019", "pfrl_2020", "diamond"]


class WrapperConfig:
    config = DEFAULT_CONFIG

    def from_kwargs(self, **kwargs) -> Callable[[Any], Any]:
        self.config = merge_dicts(self.config, kwargs)
        return self.get_wrapper()

    def from_config(self, config_file) -> Callable[[Any], Any]:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.config = merge_dicts(self.config, config)
        return self.get_wrapper()

    def get_wrapper(self) -> Callable[[Any], Any]:
        self.validate_config()
        return lambda env: wrap_env(env, self.config)

    def validate_config(self):
        count = sum([1 for option in CONFLICTING_OPTIONS if self.config[option]])
        if count == 0:
            print("No option selected!")
            print("Defaulting to pfrl 2020 wrapper settings...")
            print(
                "Warning: Default settings may change in the future! "
                "Please specify your own working wrapper configuration!"
            )
            self.config["pfrl_2020"] = True
            if self.config["pfrl_2020_config"]["action_choices"] is None:
                print(
                    "Using pre-generated kmeans actions since action_choices is not specified."
                )
                self.config["pfrl_2020_config"]["action_choices"] = load_means()
        if count > 1:
            raise ValueError(
                f"Too many options specified! "
                f"Please choose only one from {CONFLICTING_OPTIONS}."
            )
        if self.config["pfrl_2020"]:
            action_choices = self.config["pfrl_2020_config"]["action_choices"]
            if isinstance(action_choices, str) and os.path.exists(action_choices):
                action_choices = load_means(action_choices)
                self.config["pfrl_2020_config"]["action_choices"] = action_choices
            if not (action_choices is None or isinstance(action_choices, np.ndarray)):
                raise ValueError("action_choices must be None or a numpy array!")
        if self.config["diamond"]:
            action_choices = self.config["diamond_config"]["action_choices"]
            if isinstance(action_choices, str) and action_choices == "debug":
                self.config["diamond_config"]["action_choices"] = load_means()


def wrap_env(env, config):
    if config["pfrl_2019"]:
        pfrl_config = config["pfrl_2019_config"]
        return pfrl_2019_wrap_env(
            env,
            pfrl_config["test"],
            get_env_id(env),
            pfrl_config["monitor"],
            pfrl_config["outdir"],
            pfrl_config["frame_skip"],
            pfrl_config["gray_scale"],
            pfrl_config["frame_stack"],
            pfrl_config["disable_action_prior"],
            pfrl_config["always_keys"],
            pfrl_config["reverse_keys"],
            pfrl_config["exclude_keys"],
            pfrl_config["exclude_noop"],
            pfrl_config["randomize_action"],
            pfrl_config["eval_epsilon"],
        )
    elif config["pfrl_2020"]:
        pfrl_config = config["pfrl_2020_config"]
        return pfrl_2020_wrap_env(
            env,
            pfrl_config["test"],
            pfrl_config["monitor"],
            pfrl_config["outdir"],
            pfrl_config["frame_skip"],
            pfrl_config["gray_scale"],
            pfrl_config["frame_stack"],
            pfrl_config["randomize_action"],
            pfrl_config["eval_epsilon"],
            pfrl_config["action_choices"],
        )
    elif config["diamond"]:
        diamond_config = config["diamond_config"]
        return diamond_wrap_env(env, **diamond_config)
    raise NotImplementedError("No wrapper configuration detected.")
