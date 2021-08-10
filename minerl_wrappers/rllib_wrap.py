from .rllib.action_repeat_wrapper import MineRLActionRepeat
from .rllib.action_wrapper import MineRLActionWrapper
from .rllib.deterministic_wrapper import MineRLDeterministic
from .rllib.discrete_action_wrapper import (
    MineRLDiscreteActionWrapper,
)
from .rllib.gray_scale_wrapper import MineRLGrayScale
from .rllib.normalize import (
    MineRLNormalizeObservationWrapper,
    MineRLNormalizeActionWrapper,
    MineRLRewardScaleWrapper,
)
from .rllib.observation_stack_wrapper import MineRLObservationStack
from .rllib.observation_wrapper import (
    MineRLObservationWrapper,
    MineRLRemoveVecObservationWrapper,
    MineRLPOVChannelsLastWrapper,
)


def wrap_env(
    env,
    num_stack=1,
    action_repeat=1,
    gray_scale=False,
    seed=None,
    normalize_observation=False,
    normalize_action=False,
    reward_scale=1.0,
    action_choices=None,
    remove_vec_obs=False,
    channels_last=True,
    **kwargs
):
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
    discrete = False
    if action_choices is not None:
        discrete = True
        env = MineRLDiscreteActionWrapper(env, action_choices=action_choices)
    if gray_scale:
        env = MineRLGrayScale(env)
    if normalize_observation:
        env = MineRLNormalizeObservationWrapper(
            env,
            kwargs.get("norm_pov_low", 0.0),
            kwargs.get("norm_pov_high", 1.0),
            kwargs.get("norm_vec_low", -1.0),
            kwargs.get("norm_vec_high", 1.0),
        )
    if normalize_action:
        if discrete:
            print(
                "Tried to normalize discrete actions which is not possible! "
                "Skipping the normalizing action wrapper."
            )
        else:
            env = MineRLNormalizeActionWrapper(
                env, kwargs.get("norm_act_low", -1.0), kwargs.get("norm_act_high", 1.0)
            )
    if reward_scale != 1.0:
        env = MineRLRewardScaleWrapper(env, reward_scale)
    if num_stack > 1:
        env = MineRLObservationStack(env, num_stack)
    if remove_vec_obs:
        env = MineRLRemoveVecObservationWrapper(env, kwargs.get("pov_space_index", 0))
    if channels_last:
        env = MineRLPOVChannelsLastWrapper(env)
    if action_repeat > 1:
        env = MineRLActionRepeat(env, action_repeat)
    if seed is not None:
        env = MineRLDeterministic(env, seed)
    return env
