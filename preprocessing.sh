#!/bin/bash

DATA_PATH="data/RDB7"
FULL_CSV="$DATA_PATH/raw_data/rdb7_full.csv"
FULL_XYZ="$DATA_PATH/raw_data/rdb7_full.xyz"
SAVE_DIR="$DATA_PATH/processed_data"

uv run preprocessing.py \
    --csv_file "$FULL_CSV" \
    --xyz_file "$FULL_XYZ" \
    --save_dir "$SAVE_DIR"
