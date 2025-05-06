#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_test_all_model_sizes
#SBATCH --output=%x-%j.out

uv run flow_train.py -m model.num_steps=25 model.num_samples=50 task_name=train_test_all_model_sizes experiment=flow1,flow2,flow3,flow3,flow5