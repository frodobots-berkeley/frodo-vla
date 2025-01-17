# !\bin\bash
sudo su
su noam
cd ~/bigvision-palivla
source ~/.local/bin/env
source .venv/bin/activate
uv run wandb login $API_KEY
python scripts/train.py --config configs/nav_config.py