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

# Develop

Format your code with `poetry run black minerl_wrappers`.  

## Dependencies

Upgrade poetry packages with `poetry update`.

Generate precise requirements with `poetry export -f requirements.txt --output requirements.txt`.
