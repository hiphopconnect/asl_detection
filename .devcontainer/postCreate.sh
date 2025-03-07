# install poetry and dependencies
wget -qO - https://install.python-poetry.org | python3 - 

poetry config virtualenvs.in-project true

poetry install --no-root

poetry run pip install ipykernel
poetry run python -m ipykernel install --user --name=asl_detection --display-name "Python (Poetry)"

# install node dependencies
cd frontend 
npm install
