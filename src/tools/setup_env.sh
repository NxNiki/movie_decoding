#!/bin/bash

# Variables
PYTHON_VERSION="3.10"
POETRY_VERSION="1.8.3"
CONDA_DIR="$HOME/miniconda3"
PROFILE="$HOME/.bashrc"

# Function to install Miniconda
install_conda() {
    echo "Checking if Miniconda is installed..."
    if [ ! -d "$CONDA_DIR" ]; then
        echo "Miniconda is not installed. Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        bash ~/miniconda.sh -b -p $CONDA_DIR
        rm ~/miniconda.sh

        # Add conda to PATH
        echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> $PROFILE

        # Initialize conda
        source $PROFILE
        conda init
        echo "Miniconda installed successfully."
    else
        echo "Miniconda is already installed."
    fi
}

# Function to create and activate a Python 3.10 Conda environment
install_python_conda() {
    echo "Checking if Python $PYTHON_VERSION environment is installed..."
    if ! conda env list | grep "python-$PYTHON_VERSION" > /dev/null 2>&1; then
        echo "Creating a new Conda environment with Python $PYTHON_VERSION..."
        conda create -y -n python-$PYTHON_VERSION python=$PYTHON_VERSION
        echo "Conda environment with Python $PYTHON_VERSION created."
    else
        echo "Conda environment with Python $PYTHON_VERSION already exists."
    fi
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

# Install Miniconda
install_conda

# Create and activate a Python 3.10 environment using Conda
install_python_conda

# Install Poetry
install_poetry

echo "Setup complete. Python 3.10 (via Conda) and Poetry have been installed (if not already present)."
