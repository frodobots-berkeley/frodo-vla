# Quickstart for Nav-VLA finetuning

Note that the script to run on tpus will automatically push your code. If you want to isolate your development please make a new branch.

To launch on a single tpu vm (v4-8)
```
bash run_cat.sh <name of tpu> <initialize (true for first job on new tpu)> <update (true if code is changed)> <wandb api key>
```
To ssh into a single tpu vm (v4-8)
```
gcloud alpha compute tpus tpu-vm ssh <name of tpu> --zone=us-central2-b
```

To launch on a pod
```
bash run_cat_pod.sh <name of pod> <initialize (true for first job on new tpu)> <update (true if code is changed)> <wandb api key>
```
To ssh into a pod 
```
bash ssh_pod.sh <name of pod>
```

Note that on initialization of a new pod or tpu vm, you will need to login to hugging face (to be fixed). To do this
```
huggingface-cli login
```
and input your api key.

# PaliVLA
This is a framework for training multimodal vision-language-action (VLA) model for robotics in JAX. It primarily supports PaliGemma for now, though more base models will be added in the future.

## Installation
We develop with `uv`, but other environment managers should work fine. To install the dependencies, run:
```bash
uv init --python=python3.10
uv sync
```

We require some extra packages, namely [`dlimp`](https://github.com/kvablack/dlimp) and [`octo`](https://github.com/octo-models/octo) for dataloading. To install them, clone them locally. You will have to install `dlimp` without its dependencies, as its specified tensorflow version has some mutual incompatibilities with other packages we use.

## Training
To train a model, run:
```bash
python -m palivla/train.py --config_file palivla/configs/bridge_config.py
```

This repository is (for now) a fork of [`big_vision`](https://github.com/google-research/big_vision).
