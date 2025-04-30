#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_ablate_steps
#SBATCH --output=%x-%j.out

cd /home/leonard.galustian/projects/fm-gotennet || exit

mamba activate gotennet

MODEL_PATH="logs/train_test_error_bar/multiruns/2025-04-11_09-32-05/2/checkpoints/epoch_294.ckpt"

python flow_train.py -m model.num_steps=1,3,5,10,25,50 model.num_samples=25 task_name=test_ablate_steps train=False custom_model_weight_path=$MODEL_PATH
