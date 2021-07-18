# minerl-wrappers

`minerl-wrapper` compiles common wrappers and standardizes code for reproducibility in the MineRL environment!

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
