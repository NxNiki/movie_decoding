#!/bin/bash

# Variables
PYTHON_VERSION="3.10"
POETRY_VERSION="1.8.3"
CONDA_DIR="$HOME/miniconda3"
PROFILE="$HOME/.bashrc"
ENVIRONMENT_NAME="movie_decoding"
ENVIRONMENT_PATH="/u/project/ifried/data/conda_envs/"


# Function to create and activate a Python Conda environment
install_python_conda() {
    echo "Checking if environment: $ENVIRONMENT_NAME is installed..."
    if ! conda env list | grep $ENVIRONMENT_NAME > /dev/null 2>&1; then
        echo "Creating a new Conda environment: $ENVIRONMENT_NAME with Python $PYTHON_VERSION..."
        # conda create -y -n $ENVIRONMENT python=$PYTHON_VERSION
        conda create --prefix ${ENVIRONMENT_PATH}${ENVIRONMENT_NAME} python=$PYTHON_VERSION

        echo "Conda environment $ENVIRONMENT_NAME created."
    else
        echo "Conda environment $ENVIRONMENT_NAME already exists."
    fi
    conda init bash
    conda activate ${ENVIRONMENT_PATH}$ENVIRONMENT_NAME
    echo "Verifying Python version in $ENVIRONMENT_NAME..."
    python --version
}

# Function to install Poetry
install_poetry() {
    echo "Checking if Poetry is installed..."
    if ! command -v poetry > /dev/null 2>&1; then
        echo "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION

        # Add Poetry to bashrc
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> $PROFILE

        # Reload bashrc to apply changes
        source $PROFILE
        echo "Poetry installed successfully."
    else
        echo "Poetry is already installed."
    fi
}

# load the job environment:
. /u/local/Modules/default/init/modules.sh
# To see which versions of anaconda are available use: module av anaconda
module load anaconda3
echo "Loaded anaconda version:"
conda info
which conda

# Create and activate a Python environment using Conda
install_python_conda

# Install Poetry
install_poetry

echo "Setup complete. Python (via Conda) and Poetry have been installed."
