from typing import Callable, Any

import numpy as np

from .pfrl_wrappers import wrap_env as pfrl_2020_wrap_env
from .utils import merge_dicts

DEFAULT_CONFIG = {
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
    },
}


class WrapperConfig:
    config = DEFAULT_CONFIG

    def from_kwargs(self, **kwargs) -> Callable[[Any], Any]:
        self.config = merge_dicts(self.config, kwargs)
        return self.get_wrapper()

    def from_config(self, config_file) -> Callable[[Any], Any]:
        raise NotImplementedError

    def get_wrapper(self) -> Callable[[Any], Any]:
        self.validate_config()
        return lambda env: wrap_env(env, self.config)

    def validate_config(self):
        if self.config["pfrl_2020"]:
            action_choices = self.config["pfrl_2020_config"]["action_choices"]
            if action_choices is None or not isinstance(action_choices, np.ndarray):
                raise ValueError("action_choices must be a numpy array!")


def wrap_env(env, config):
    if config["pfrl_2020"]:
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
    raise NotImplementedError()
