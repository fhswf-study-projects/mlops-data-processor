# docker-container-template
Template repository for easier definition of new docker containers for apps.

Please remind to locking your dependencies via `poetry.lock`. If your app needs `main.py` or similar, feel free to create it.

Please be sure before starting the development:
1. Install needed development dependencies via `poetry install --with dev`
2. Run `pre-commit install`. This will install the hooks in your `.git` local folder
