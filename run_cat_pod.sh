#!/bin/bash

# Check if a TPU VM name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <tpu-vm-name>"
    exit 1
fi

TPU_VM_NAME=$1
INIT=$2
UPDATE=$3
PROJECT="vlm-guidance"
ZONE="us-central2-b"
API_KEY=$4

echo "Update? $UPDATE"
echo "Init? $INIT"

TPU_INFO=$(gcloud compute tpus tpu-vm describe $TPU_VM_NAME --project=$PROJECT --zone=$ZONE --format=json 2>/dev/null)
IP = $(echo "$TPU_INFO" | jq '.networkEndpoints[0].ipAddress')
echo "TPU_INFO: $TPU_INFO"

# Cache file for TPU name/zone mapping
CACHE_FILE="$HOME/.cache/tpus"
mkdir -p "$(dirname "$CACHE_FILE")"

echo "TPU_VM_NAME: $TPU_VM_NAME"
echo "ZONE: $ZONE"
echo "Number of workers: $NUM_WORKERS"

# Copy the source directory to the TPU VM
if $INIT; then
    echo "Copying source directory to TPU VM"
    gcloud alpha compute tpus tpu-vm scp ~/.ssh/id_ed25519 $TPU_VM_NAME: --zone=us-central2-b --worker=all
    gcloud alpha compute tpus tpu-vm scp tpu_config_cat.sh $TPU_VM_NAME: --zone=us-central2-b --worker=all
    gcloud alpha compute tpus tpu-vm scp update.sh $TPU_VM_NAME: --zone=us-central2-b --worker=all
    gcloud alpha compute tpus tpu-vm scp launch.sh $TPU_VM_NAME:/tmp/launch.sh --zone=us-central2-b --worker=all
 
    echo "Initializing TPU VM"
    gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME --zone=us-central2-b --command="bash tpu_config_cat.sh" --worker=all
fi 

if $UPDATE; then
    echo "Updating source directory on TPU VM"
    git add *
    git commit -m "Update"
    git push
    gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME --zone=us-central2-b --command="bash update.sh" --worker=all
fi

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME --zone=us-central2-b --worker=all --command='tmux kill-session -t l; tmux new -s l -d "bash -li /tmp/launch.sh $API_KEY"'