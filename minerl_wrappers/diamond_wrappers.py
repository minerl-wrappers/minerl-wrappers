from .core.action_repeat_wrapper import MineRLActionRepeat
from .core.action_wrapper import MineRLFlattenActionWrapper
from .core.deterministic_wrapper import MineRLDeterministic
from .core.discrete_action_wrapper import (
    MineRLDiscreteActionWrapper,
)
from .core.gray_scale_wrapper import MineRLGrayScale
from .core.normalize import (
    MineRLNormalizeObservationWrapper,
    MineRLNormalizeActionWrapper,
    MineRLRewardScaleWrapper,
)
from .core.observation_stack_wrapper import MineRLObservationStack
from .core.observation_wrapper import (
    MineRLTupleObservationWrapper,
    MineRLRemoveVecObservationWrapper,
    MineRLPOVChannelsFirstWrapper,
)


def wrap_env(
    env,
    frame_stack=1,
    frame_skip=1,
    gray_scale=False,
    seed=None,
    normalize_observation=False,
    normalize_action=False,
    reward_scale=1.0,
    action_choices=None,
    include_vec_obs=True,
    channels_first=False,
    tuple_obs_space=True,
    flatten_action_space=True,
    **kwargs
):
    if not include_vec_obs:
        tuple_obs_space = True
    if tuple_obs_space:
        env = MineRLTupleObservationWrapper(env)
    if flatten_action_space:
        env = MineRLFlattenActionWrapper(env)
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
    if not include_vec_obs:
        env = MineRLRemoveVecObservationWrapper(env, kwargs.get("pov_space_index", 0))
    if channels_first:
        env = MineRLPOVChannelsFirstWrapper(env)
    if reward_scale != 1.0:
        env = MineRLRewardScaleWrapper(env, reward_scale)
    if frame_stack > 1:
        env = MineRLObservationStack(env, frame_stack)
    if frame_skip > 1:
        env = MineRLActionRepeat(env, frame_skip)
    if seed is not None:
        env = MineRLDeterministic(env, seed)
    return env
