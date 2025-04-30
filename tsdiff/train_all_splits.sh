#!/bin/bash
#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=train_tsdiff
#SBATCH --array=0-2


cd /home/leonard.galustian/projects/tsdiff-master || exit

mamba activate tsdiff

splits=(
    "data/RDB7/splits/rxn_core_split.pkl"
    "data/RDB7/splits/barrier_split.pkl"
    "data/RDB7/splits/random_split.pkl"
)

# Use SLURM_ARRAY_TASK_ID to select the current split
current_split="${splits[$SLURM_ARRAY_TASK_ID]}"

splitname=$(basename "$current_split")
splitname="${splitname%.pkl}"

echo "Running training with split ${current_split} (splitname: ${splitname})"

python train.py \
    ./configs/train_config_v2.yml \
    --split_file "${current_split}" \
    --logdir ./logs/trained_rdb7/ \
    --project tsdiff-rdb7 \
    --name "goflow-train-split-${splitname}"
