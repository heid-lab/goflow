#!/usr/bin/env python3

"""
Script: find_reaction_types.py

Reads a CSV with classified reaction data, groups reactions by a user-specified column
(which can be either the column's name or its zero-based index), and outputs a CSV
with reaction types, each associated with a list of reaction IDs.

Usage:
    python find_reaction_types.py input.csv group_by_column
        [--out_file reaction_types.csv]
        [--log_file reaction_types.log]
"""

import sys
import logging
import argparse
import pandas as pd
from pathlib import Path

def configure_logging(log_file: Path) -> None:
    """
    Configure the Python logging system to write INFO-level logs to the given file.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',   # overwrite each time
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the reaction type identification script.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Identify reaction types by grouping reactions based on a specified column."
    )
    # Positional arguments
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV containing classified reactions."
    )
    parser.add_argument(
        "group_by_column",
        type=str,
        help="Column name or zero-based index to group by (e.g. 'reaction_template' or '3')."
    )
    # Optional arguments
    parser.add_argument(
        "--out_file",
        type=str,
        default="reaction_types.csv",
        help="Output CSV name (default: reaction_types.csv)."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="reaction_types.log",
        help="Log file name (default: reaction_types.log)."
    )
    return parser.parse_args()

def interpret_column_spec(df: pd.DataFrame, column_spec: str) -> str:
    """
    Interpret 'column_spec' as either a zero-based column index or a column name.
    Returns the actual column name from the DataFrame if valid, or raises ValueError.
    """
    if column_spec.isdigit():
        col_idx = int(column_spec)
        if col_idx < 0 or col_idx >= df.shape[1]:
            raise ValueError(
                f"Column index {col_idx} is out of range. "
                f"DataFrame has {df.shape[1]} columns (0..{df.shape[1]-1})."
            )
        return df.columns[col_idx]
    else:
        if column_spec not in df.columns:
            raise ValueError(
                f"Column '{column_spec}' not found in DataFrame columns: {list(df.columns)}"
            )
        return column_spec

def identify_reaction_types(
    input_csv: Path,
    group_by_column: str,
    output_csv: Path,
    log_file: Path) -> None:

    configure_logging(log_file)
    logging.info("Starting reaction type identification script.")
    logging.info(f"Reading input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} rows.")

    actual_column_name = interpret_column_spec(df, group_by_column)
    logging.info(f"Grouping reactions by column: '{actual_column_name}'")

    if actual_column_name not in df.columns:
        logging.error(f"Column '{actual_column_name}' not found in input CSV.")
        sys.exit(1)

    grouped = df.groupby(actual_column_name)

    out_columns = ["reaction_type_index", "reaction_type", "reaction_indices"]
    out_df = pd.DataFrame(columns=out_columns)

    for idx, (key, group) in enumerate(grouped, start=1):
        reaction_ids = group.iloc[:, 0].tolist()
        # Build a single-row DataFrame to concat
        new_row = pd.DataFrame([{
            "reaction_type_index": idx,
            "reaction_type": key,
            "reaction_indices": reaction_ids
        }])
        out_df = pd.concat([out_df, new_row], ignore_index=True)

    out_df["num_associated_reactions"] = out_df["reaction_indices"].apply(len)
    out_df = out_df.sort_values(by="num_associated_reactions", ascending=False).reset_index(drop=True)
    out_df["reaction_type_index"] = range(1, len(out_df) + 1)
    out_df.drop(columns=["num_associated_reactions"], inplace=True)

    logging.info(f"Encountered {len(out_df)} unique reaction types.")
    logging.info(f"Writing output CSV to {output_csv}")
    out_df.to_csv(output_csv, index=False)
    logging.info("Reaction type identification complete.")
    logging.info(f"\nDone!")

def main() -> None:
    """
    Main entry point: parse args, run reaction type identification.
    """
    args = parse_arguments()
    input_csv = Path(args.input_csv)
    group_by_column = args.group_by_column
    output_csv = Path(args.out_file)
    log_file = Path(args.log_file)

    identify_reaction_types(input_csv, group_by_column, output_csv, log_file)

if __name__ == "__main__":
    main()
