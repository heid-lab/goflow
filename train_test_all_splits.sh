#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_test_all_splits
#SBATCH --output=%x-%j.out

cd /home/leonard.galustian/projects/fm-gotennet || exit

mamba activate gotennet

python flow_train.py -m \
    data.split_file="rxn_core_split.pkl","barrier_split.pkl","random_split.pkl" \
    model.num_steps=25 \
    model.num_samples=25 \
    task_name=train_test_all_splits
