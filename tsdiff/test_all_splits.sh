#!/bin/bash

#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --nodes=1
#SBATCH --job-name=test_tsdiff
#SBATCH --array=0-2

cd /home/leonard.galustian/projects/tsdiff-master || exit

mamba activate tsdiff

SPLIT_WEIGHTS=(
    "logs/trained_rdb7/train_config_v2_2025_04_11__10_39_53_goflow-train-split-rxn_core_split/checkpoints/29801.pt"
    "logs/trained_rdb7/train_config_v2_2025_04_11__11_48_05_goflow-train-split-barrier_split/checkpoints/21401.pt"
    "logs/trained_rdb7/train_config_v2_2025_04_11__12_39_03_goflow-train-split-random_split/checkpoints/33001.pt"
)

SPLITS=(
    "data/RDB7/splits/rxn_core_split.pkl"
    "data/RDB7/splits/barrier_split.pkl"
    "data/RDB7/splits/random_split.pkl"
)

SAVE_PATH="reproduce/RDB7"
DATA_PATH="data/RDB7/processed_data/data.pkl"

# Use SLURM_ARRAY_TASK_ID to index the arrays
IDX=${SLURM_ARRAY_TASK_ID}
CURRENT_SPLIT="${SPLITS[$IDX]}"
CURRENT_SPLIT_WEIGHTS="${SPLIT_WEIGHTS[$IDX]}"
SPLIT_NAME=$(basename "$CURRENT_SPLIT" .pkl)

# Run sampling
echo "Running sampling for split ${CURRENT_SPLIT} (weights: ${CURRENT_SPLIT_WEIGHTS})"
python sampling.py \
    "$CURRENT_SPLIT_WEIGHTS" \
    --sampling_type ld \
    --save_dir "$SAVE_PATH" \
    --batch_size 200 \
    --data_path "$DATA_PATH" \
    --test_split "$CURRENT_SPLIT"

# Run evaluation
echo "Running evaluation for split ${CURRENT_SPLIT} (split name: ${SPLIT_NAME})"
python evaluation.py \
    --pred_data "${SAVE_PATH}/samples_${SPLIT_NAME}.pkl" \
    --save_path "$SAVE_PATH" \
    --split_name "$SPLIT_NAME"
