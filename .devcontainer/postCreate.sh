# install poetry and dependencies
wget -qO - https://install.python-poetry.org | python3 - 
poetry install --no-root

# install node dependencies
cd frontend 
npm install
