#!/bin/bash
# Activate ssh key 
mkdir -p /home/$USER/.ssh
mv id_ed25519 /home/$USER/.ssh
eval "$(ssh-agent -s)"
ssh-add id_ed25519
touch /home/$USER/.ssh/known_hosts
ssh-keyscan github.com >> /home/$USER/.ssh/known_hosts
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
git clone https://github.com/catglossop/bigvision-palivla.git --recursive
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
uv python pin 3.11.12
uv venv --python=python3.11.12
uv sync --extra tpu  

# For inference
uv pip install google-cloud-logging
uv pip install google-cloud-storage