#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_test_error_bar
#SBATCH --output=%x-%j.out

uv run flow_train.py -m seed=1,2,3,4,5,6,7,8 model.num_steps=25 model.num_samples=25 task_name=train_test_error_bar
