# color_picker

Steps:
1 - Install pyenv-win https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md#pyenv-win-zip

2 - pip install pipenv

3 - in the fastapi directory run: pipenv install --dev

4 - Run pipenv shell

5 - uvicorn recommender_system.api:app

use the following command in terminal to test if it works or not:

curl -X POST -H "Content-Type: application/json" --data "{\"url\": \"https://dl5zpyw5k3jeb.cloudfront.net/photos/pets/56684616/1/\"}" http://127.0.0.1:8000/identify_colors

# What is this?
Color Picker is an AI feature for a pet adoption website that helps in identifying the pet colors exact name by:

1 - Removing Background from the Image

2 - Keep Focus to the pet

3 - Extract main dominant colors with percentages

4 - Use Machine Learning Algorithms to classify the exact color name


## Preview

![background-removal](https://github.com/user-attachments/assets/a40be199-bc3c-4a19-b95f-22f5f0e774de)


