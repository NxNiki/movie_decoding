#!/bin/bash

# Exit the script if any command fails
set -e

# Variables
PYTHON_VERSION="3.10"
POETRY_VERSION="1.8.3"
PROFILE="$HOME/.bashrc"
ENVIRONMENT_NAME="movie_decoding"
POETRY_ENV_DIR="$HOME/.poetry_env"

# Trap to handle errors and ensure environment deactivation
trap 'echo "Error occurred. Exiting script."; deactivate 2>/dev/null || true; exit 1;' ERR

# Function to create and activate a Python Conda environment
install_python_conda() {
    echo "Checking if environment: $ENVIRONMENT_NAME is installed..."
    if ! conda env list | grep $ENVIRONMENT_NAME > /dev/null 2>&1; then
        echo "Creating a new Conda environment: $ENVIRONMENT_NAME with Python $PYTHON_VERSION..."
        conda create -y -n $ENVIRONMENT_NAME python=$PYTHON_VERSION || { echo "Failed to create Conda environment."; exit 1; }

        echo "Conda environment $ENVIRONMENT_NAME created."
    else
        echo "Conda environment $ENVIRONMENT_NAME already exists."
    fi
    conda init bash
    conda activate $ENVIRONMENT_NAME || { echo "Failed to activate Conda environment."; exit 1; }
    echo "Verifying Python version in $ENVIRONMENT_NAME..."
    python --version || { echo "Python version verification failed."; exit 1; }
}

install_poetry() {
# Install Poetry in a dedicated virtual environment on an SGE server

    echo "Checking if Poetry is installed in the virtual environment..."
    
    # Check if the virtual environment exists
    if [ ! -d "$POETRY_ENV_DIR" ]; then
        echo "Creating a dedicated virtual environment for Poetry..."

        # Create the virtual environment
        python3 -m venv "$POETRY_ENV_DIR" || { echo "Failed to create virtual environment for Poetry."; exit 1; }
    fi

    # Activate the virtual environment
    source "$POETRY_ENV_DIR/bin/activate" || { echo "Failed to activate virtual environment."; exit 1; }

    # Check if Poetry is installed by trying to run 'poetry --version'
    if ! command -v poetry > /dev/null 2>&1; then
        echo "Poetry is not installed. Installing Poetry..."
        # curl -sSL https://install.python-poetry.org | python3 - || { echo "Failed to install Poetry."; exit 1; }

        pip install poetry==$POETRY_VERSION

        # Add Poetry's virtual environment to the PATH in your profile
        # echo 'export PATH="$HOME/.poetry_env/bin:$PATH"' >> $PROFILE || { echo "Failed to update PATH."; exit 1; }

        # Reload the profile to apply changes
        source $PROFILE || { echo "Failed to reload profile."; exit 1; }

        echo "Poetry installed successfully in the virtual environment."
    else
        # If Poetry is installed, check if it's the correct version
        INSTALLED_POETRY_VERSION=$(poetry --version | awk '{print $3}')
        if [ "$INSTALLED_POETRY_VERSION" != "$POETRY_VERSION" ]; then
            echo "Poetry version mismatch. Expected: $POETRY_VERSION, Installed: $INSTALLED_POETRY_VERSION"
            echo "Updating Poetry to version $POETRY_VERSION..."
            poetry self update --version $POETRY_VERSION || { echo "Failed to update Poetry."; exit 1; }
        else
            echo "Poetry is already installed and up to date (version $INSTALLED_POETRY_VERSION)."
        fi
    fi

    # Deactivate the virtual environment
    deactivate
    echo "Virtual environment deactivated."
}

# Load the modules initialization script
. /u/local/Modules/default/init/modules.sh

# Load the Python module
echo "Loading the required Python module..."
module load python/3.9.6 || { echo "Failed to load Python module."; exit 1; }

# Install Poetry
# make sure poetry is not installed in any conda environment
install_poetry

# Load the Anaconda module
echo "Loading the required Anaconda module..."
module load anaconda3/2023.03 || { echo "Failed to load Anaconda module."; exit 1; }

echo "Loaded Anaconda version:"
which conda || { echo "Failed to locate Conda."; exit 1; }

# Create and activate a Python environment using Conda
install_python_conda

echo "Setup complete. Python (via Conda) and Poetry have been installed."
