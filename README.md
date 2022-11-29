# color_picker

Steps:
1 - Install pyenv-win https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md#pyenv-win-zip
2 - pip install pipenv
3 - in the fastapi directory run: pipenv install --dev
4 - Run pipenv shell
5 - uvicorn recommender_system.api:app

use the following command in terminal to test if it works or not:
curl -X POST -H "Content-Type: application/json" --data "{\"url\": \"https://dl5zpyw5k3jeb.cloudfront.net/photos/pets/56684616/1/\"}" http://127.0.0.1:8000/identify_colors

