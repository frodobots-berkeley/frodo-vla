#! /bin/bash


TPU_VM_NAME=$1
# If updating repo, pull 
git add *
git commit -m "Update"
git push
gcloud alpha compute tpus tpu-vm scp run_on_tpu.sh $TPU_VM_NAME: --zone=us-central2-b
gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME --zone=us-central2-b
