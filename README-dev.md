# Develop

For development, I highly recommend using `poetry` to install your dev environment:
```
poetry install
```

## Formatting
Format your code with `poetry run black minerl_wrappers`.  

## Dependencies

Upgrade poetry packages with `poetry update`.  
If anyone else adds requirements run `poetry install` to update your local environment.  
You can add dependency packages with `poetry add <package-name>`.  
Make sure to commit `poetry.lock` and `pyproject.toml` after making any changes!  
Lastly, generate precise requirements with `poetry export -f requirements.txt --output requirements.txt`.

## Testing

Run tests with `poetry run pytest tests`.
For code coverage, run `poetry run pytest tests --cov=minerl_wrappers`.

## Pull Requests
When you make a pull request, automated testing checks will run.
In order to merge, make sure to get at least one review from a contributor and that your tests pass.
Tests can fail either because tests didn't pass, the code wasn't formatted correctly, or no tests were added to maintain code coverage.
