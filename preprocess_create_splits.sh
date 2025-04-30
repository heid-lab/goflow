#!/bin/bash

# Create dataset splits

DATA_PATH="data/RDB7"
FULL_CSV="$DATA_PATH/raw_data/rdb7_full.csv"
RXN_CORE_CSV="$DATA_PATH/raw_data/reaction_types.csv"

python split_preprocessed.py \
    --input_rxn_csv "$FULL_CSV" \
    --output_rxn_indices_path "$DATA_PATH/splits" \
    --random \
    --rxn_core_clusters \
    --rxn_core_clusters_csv "$RXN_CORE_CSV" \
    --barrier_height