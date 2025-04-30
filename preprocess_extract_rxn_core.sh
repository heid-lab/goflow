#!/bin/bash

# Create reaction core dataset split.

DATA_PATH="data/RDB7"
FULL_CSV="$DATA_PATH/raw_data/rdb7_full.csv"
RXNS_CLS_CSV="$DATA_PATH/raw_data/rxn_core_classified.csv"
RXN_TYPES_CSV="$DATA_PATH/raw_data/reaction_types.csv"

python data_processing/classify_by_reaction_core.py \
    "$FULL_CSV" \
    --out_file "$RXNS_CLS_CSV" \
    --log_file "$DATA_PATH/logs/classification.log"

python data_processing/find_reaction_types.py \
    "$RXNS_CLS_CSV" \
    reaction_template \
    --out_file "$RXN_TYPES_CSV" \
    --log_file "$DATA_PATH/logs/reaction_types.log"
