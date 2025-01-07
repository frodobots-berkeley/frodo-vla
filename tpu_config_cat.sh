#!/bin/bash

# Install conda
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda init bash

# Install octo dependencies
conda create -n bigvision python=3.10
conda activate bigvision

# Activate ssh key 
mkdir -p /home/noam/.ssh
mv id_ed25519 /home/noam/.ssh
eval "$(ssh-agent -s)"
ssh-add id_ed25519
touch /home/noam/.ssh/known_hosts
ssh-keyscan github.com >> /home/noam/.ssh/known_hosts

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
git clone git@github.com:catglossop/bigvision-palivla.git --recursive
cd ~/bigvision-palivla
uv venv --python=python3.11
uv sync --extra tpu