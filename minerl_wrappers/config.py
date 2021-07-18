from typing import Callable, Any

from pfrl_wrappers import wrap_env as pfrl_wrap_env
from utils import merge_dicts

DEFAULT_CONFIG = {
    "pfrl": False,
    "pfrl_config": {
        "test": False,
        "monitor": False,
        "outdir": "results",
        "frame_skip": None,
        "gray_scale": False,
        "randomize_action": False,
        "eval_epsilon": 0.001,
        "action_choices": None,
    }
}


class WrapperConfig:
    config = DEFAULT_CONFIG

    def from_kwargs(self, **kwargs) -> Callable[[Any], Any]:
        self.config = merge_dicts(self.config, kwargs)
        return self.get_wrapper()

    def from_config(self, config_file) -> Callable[[Any], Any]:
        raise NotImplementedError

    def get_wrapper(self) -> Callable[[Any], Any]:
        return lambda env: wrap_env(env, self)


def wrap_env(env, wc_config: WrapperConfig):
    config = wc_config.config
    if config["pfrl"]:
        pfrl_config = config["pfrl_config"]
        return pfrl_wrap_env(
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
    return NotImplementedError
