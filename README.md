# minerl-wrappers

![Tests](https://github.com/minerl-wrappers/minerl-wrappers/actions/workflows/workflow.yaml/badge.svg)
[![codecov](https://codecov.io/gh/minerl-wrappers/minerl-wrappers/branch/main/graph/badge.svg?token=0BPxXISonq)](https://codecov.io/gh/minerl-wrappers/minerl-wrappers)
![PyPI](https://img.shields.io/pypi/v/minerl-wrappers)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/minerl-wrappers)
![Libraries.io dependency status for latest release](https://img.shields.io/librariesio/release/pypi/minerl-wrappers)

`minerl-wrappers` compiles common wrappers and standardizes code for reproducibility in the [MineRL environment](https://minerl.readthedocs.io/en/latest/index.html)!

# Currently Supported Environments
- MineRL Basic Environments
  - [`MineRLTreechop-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerltreechop-v0)
  - [`MineRLNavigate-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigate-v0)
  - [`MineRLNavigateDense-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigatedense-v0)
  - [`MineRLNavigateExtreme-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigateextreme-v0)
  - [`MineRLNavigateExtremeDense-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlnavigateextremedense-v0)
  - [`MineRLObtainDiamond-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtaindiamond-v0)
  - [`MineRLObtainDiamondDense-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtaindiamonddense-v0)
  - [`MineRLObtainIronPickaxe-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtainironpickaxe-v0)
  - [`MineRLObtainIronPickaxeDense-v0`](https://minerl.readthedocs.io/en/latest/environments/index.html#minerlobtainironpickaxedense-v0)
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
and [2019 PfN minerl baselines](https://github.com/minerllabs/baselines/tree/master/2019)
  - Supports Basic Environments for 2019 and Diamond Competition Environments for 2020
- diamond wrappers: updated wrappers for the 2021 MineRL Diamond Competition Environments

## Wrap arguments
For documentation see wrapper files:  
[pfrl_2019_wrappers.py](https://github.com/minerl-wrappers/minerl-wrappers/blob/main/minerl_wrappers/pfrl_2019_wrappers.py)  
[pfrl_2020_wrappers.py](https://github.com/minerl-wrappers/minerl-wrappers/blob/main/minerl_wrappers/pfrl_2020_wrappers.py)
[diamond_wrappers.py](https://github.com/minerl-wrappers/minerl-wrappers/blob/main/minerl_wrappers/diamond_wrappers.py)

# Requirements
- Java JDK 8
- Python 3.7+
- `minerl==0.4.1`

# Install

Make sure you have Java JDK 8 installed as the only Java version for MineRL.

Install with pip from pypi:
```bash
pip install minerl-wrappers
```

Install directly from git:
```bash
pip install git+https://github.com/minerl-wrappers/minerl-wrappers.git
```

## Clone and Install
```bash
git clone https://github.com/minerl-wrappers/minerl-wrappers.git
cd minerl-wrappers
```

### Use your own virtual environment

#### virtualenv
Installed Python 3.7+
```bash
python3 -m virtualenv venv
source venv/bin/activate
```

#### conda
Install Anaconda or Miniconda
```bash
conda create --name minerl-wrappers python=3.7
conda activate minerl-wrappers
```

### Install dependencies
1. Install dependencies with pip:
  ```bash
  # install fixed requirements
  pip install -r requirements.txt
  # set the minerl-wrappers module for imports
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  ```
2. Install dependencies with [`poetry`](https://python-poetry.org/docs/#installation) into your virtual environment:
  ```bash
  # this also installs minerl-wrappers as a package
  poetry install --no-dev
  ```

# Use

To quickly test out the wrappers try:
```python
import gym
import minerl
from minerl_wrappers import wrap

env = gym.make("MineRLObtainDiamondDenseVectorObf-v0")
env = wrap(env) # plug this into your rl algorithm
```

Change which wrappers to apply by supplying config arguments:
```python
config = {
  "diamond": True,
  "diamond_config": {
    "frame_skip": 4,
    "frame_stack": 4,
  }
}
env = wrap(env, **config)
```

# Contributing
It is highly encouraged to contribute wrappers that worked well for you!
Just create a [Pull Request](https://github.com/minerl-wrappers/minerl-wrappers/pulls) on this repository, 
and we'll work together to get it merged!
Read [`README-DEV.md`](https://github.com/minerl-wrappers/minerl-wrappers/blob/main/README-DEV.md) for contributing guidelines and more details!
