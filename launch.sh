API_KEY=$1

cd ~/bigvision-palivla
source ~/.local/bin/env
source .venv/bin/activate
huggingface-cli login
uv run wandb login $API_KEY
python scripts/train.py --config configs/nav_config.py
