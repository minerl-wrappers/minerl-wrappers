from .config import WrapperConfig


def wrap(env, config_file=None, **kwargs):
    wrapper_config = WrapperConfig()
    if config_file is not None:
        wrapper_fn = wrapper_config.from_config(config_file)
    else:
        wrapper_fn = wrapper_config.from_kwargs(**kwargs)
    return wrapper_fn(env)
