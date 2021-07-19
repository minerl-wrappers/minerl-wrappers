# minerl-wrappers

`minerl-wrapper` compiles common wrappers and standardizes code for reproducibility in the MineRL environment!

# Currently Supported Environments

- MineRL Diamond Competition Environments
  - [`MineRLTreechopVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerltreechopvectorobf-v0)
  - [`MineRLNavigateVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigatevectorobf-v0)
  - [`MineRLNavigateDenseVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigatedensevectorobf-v0)
  - [`MineRLNavigateExtremeVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigateextremevectorobf-v0)
  - [`MineRLNavigateExtremeDenseVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigateextremedensevectorobf-v0)
  - [`MineRLObtainDiamondVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtaindiamondvectorobf-v0)
  - [`MineRLObtainDiamondDenseVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtaindiamonddensevectorobf-v0)
  - [`MineRLObtainIronPickaxeVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtainironpickaxevectorobf-v0)
  - [`MineRLObtainIronPickaxeDenseVectorObf-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtainironpickaxedensevectorobf-v0)

# Wappers
- pfrl wrappers: an assortment of wrappers ported over from the [2020 PfN minerl baselines](https://github.com/minerllabs/baselines/tree/master/2020)

## Wrap arguments
- `pfrl=False`: set `True` to use the pfrl wrappers
- `pfrl_config`: dictionary configuration for pfrl wrappers
  - `test=False`: not used
  - `monitor`
  - `outdir`
  - `frame_skip`
  - `gray_scale`
  - `frame_stack`
  - `randomize_action`
  - `eval_epsilon`
  - `action_choices`

# Install

## Poetry Installation

Install [poetry](https://python-poetry.org/docs/#installation)

Make sure you have java jdk 8 installed as the only version.

To create a virtual environment with all dependencies:
```
poetry install
```

## virtualenv
Install Python 3.7+
```
virtualenv venv
source venv/bin/activate
pip install requirements.txt
```

# Use

To quickly test out the wrappers try:
```python
import gym
import minerl
from minerl_wrappers import wrap

env = gym.make("MineRLObtainDiamondDenseVectorObf-v0")
env = wrap(env)
```

Change which wrappers to apply by supplying config arguments:
```python
config = {
  "pfrl": True,
  "pfrl_config": {
    "frame_skip": 4,
    "frame_stack": 4,
  }
}
env = wrap(env, **config)
```

# Develop

Format your code with `poetry run black minerl_wrappers`.  

## Dependencies

Upgrade poetry packages with `poetry update`.

Generate precise requirements with `poetry export -f requirements.txt --output requirements.txt`.
