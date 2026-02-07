#!/bin/bash

# setup.sh - Simple Poetry setup for Network Anomaly Detection project

set -e  # Exit on error 

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed!"
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python -
    echo "Restart terminal and rerun: source ~/.bashrc"
    exit 1
fi

# Check if pyproject.toml exists
if [[ ! -f "pyproject.toml" ]]; then
    echo "pyproject.toml not found!"
    echo "Ensure this script is ran from the project root directory."
    exit 1
fi

# Configure Poetry to create virtual environment in project directory
poetry config virtualenvs.in-project true

# Install dependencies (including dev dependencies)
echo "Installing dependencies"
poetry install

# Create kernel for notebooks
poetry run ipython kernel install --name "network-anomaly-detection" --user


echo "To activate the environment in your current shell, run:"
echo "  source \$(poetry env info --path)/bin/activate"
echo ""
echo "Or use Poetry commands directly:"
echo "  poetry run python script.py    # Run Python scripts"
echo "  poetry shell                   # Start a new shell with activated environment"
echo ""

# Automatically activate the environment for this session
VENV_PATH=$(poetry env info --path)
if [[ -f "$VENV_PATH/Scripts/activate" ]]; then
    echo "Activating env (note that the setup.sh must be sourced rather than executed)"
    source "$VENV_PATH/Scripts/activate"
    echo "Environment activated! You can now use Python directly."
    echo ""
    echo "Python version: $(python --version)"
    echo "Python location: $(which python)"
    echo ""
    echo "Installed packages:"
    pip list --format=columns
else
    echo "Could not automatically activate environment."
    exit 1
fi
