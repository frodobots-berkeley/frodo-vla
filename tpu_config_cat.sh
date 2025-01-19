#!/bin/bash

# Activate ssh key 
mkdir -p /home/noam/.ssh
mv id_ed25519 /home/noam/.ssh
eval "$(ssh-agent -s)"
ssh-add id_ed25519
touch /home/noam/.ssh/known_hosts
ssh-keyscan github.com >> /home/noam/.ssh/known_hosts

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone git@github.com:catglossop/bigvision-palivla.git --recursive
cd ~/bigvision-palivla
git pull
git submodule sync --recursive
cd ~/bigvision-palivla/octo
git fetch 
git checkout origin/main
git branch main -f 
git checkout main
cd ~/bigvision-palivla
source .venv/bin/activate
uv venv --python=python3.11
uv sync --extra tpu  
