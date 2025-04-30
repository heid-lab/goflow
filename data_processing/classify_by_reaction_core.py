#!/usr/bin/env python3

"""
Script: classify_by_reaction_core.py

Reads a CSV with reaction data (first column = reaction ID, second column = 'reactant>>product' SMILES).
Extracts a small mapped reaction core, then unmaps it to produce a generalized reaction template.
Outputs a new CSV with columns:
    1) reaction_id
    2) original_smiles
    3) mapped_core
    4) unmapped_core

Usage:
    python classify_by_reaction_core.py input.csv --out_file classified.csv --log_file classification.log
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from utils.reaction_utils import *

# Disable RDKit warnings globally
RDLogger.DisableLog("rdApp.*")

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def configure_logging(log_file: Path) -> None:
    """
    Configure the Python logging system to write INFO-level logs to the given file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='w',   # overwrite each time
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# -------------------------------------------------------------------
# Argument Parsing & Main
# -------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for classification script.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Classify reactions by extracting mapped cores and unmapped templates."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV (ID in col1, 'reactant>>product' in col2)."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="classified.csv",
        help="Output CSV name (default: classified.csv)."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="classification_by_core.log",
        help="Log file name (default: classification.log)."
    )
    return parser.parse_args()

# -------------------------------------------------------------------
# Main classification function
# -------------------------------------------------------------------

def classify_by_reaction_core(
    input_csv: Path,
    output_csv: Path,
    log_file: Path
) -> None:
    """
    Classify reactions by extracting a small mapped core and then unmapping it.
    Produces a new CSV with columns:
      1) reaction_id
      2) original_smiles
      3) mapped_core
      4) unmapped_core

    Parameters
    ----------
    input_csv : Path
        The CSV file with at least 2 columns: [ID, 'reactant>>product'].
    output_csv : Path
        The CSV file to write the classification results.
    log_file : Path
        Path to the log file.

    Returns
    -------
    None
        (Writes a CSV and a log.)
    """

    # 1) Configure logging
    configure_logging(log_file)
    logging.info("Starting classification script.")
    logging.info(f"Reading input CSV: {input_csv}")

    # 2) Read input CSV
    df = pd.read_csv(input_csv)
    if df.shape[1] < 2:
        logging.error("Input CSV must have at least two columns (ID, reaction_smiles).")
        sys.exit(1)

    logging.info(f"Loaded {len(df)} rows.")

    # 3) Prepare output DataFrame
    out_columns = ["reaction_id", "original_smiles", "reaction_core", "reaction_template"]
    out_df = pd.DataFrame(columns=out_columns)

    # 4) Process each row
    for idx, row in df.iterrows():
        reaction_id = row[df.columns[0]]
        reaction_smiles = row[df.columns[1]]

        logging.info(f"Processing ID={reaction_id} with SMILES='{reaction_smiles}'")

        # Check for delimiter
        if ">>" not in reaction_smiles:
            logging.error(f"Reaction {reaction_id} has no '>>' delimiter.")
            raise ValueError(f"Reaction {reaction_id} missing '>>': {reaction_smiles}")

        r_smiles, p_smiles = reaction_smiles.split(">>")

        # 4a) Extract small mapped reaction
        small_rxn = get_small_rxn_smiles_and_mapnums2remove(r_smiles, p_smiles)
        rxn_core = small_rxn[0]
        r_template_smiles, p_template_smiles = rxn_core.split(">>")
        # mapped_rxn is e.g. "CCNC>>C=N.CC"

        # 4b) Unmap it
        r_unmapped_template_smiles = smarts2smarts(unmap_smarts(r_template_smiles))
        p_unmapped_template_smiles = smarts2smarts(unmap_smarts(p_template_smiles))
        rxn_template = r_unmapped_template_smiles + ">>" + p_unmapped_template_smiles

        # 5) Add row to output
        out_df.loc[idx] = [
            reaction_id,
            reaction_smiles,
            rxn_core,
            rxn_template
        ]

    # 6) Write output CSV
    logging.info(f"Writing output CSV to {output_csv}")
    out_df.to_csv(output_csv, index=False)
    logging.info("Classification complete.")
    logging.info(f"\nDone!")

def main() -> None:
    """
    Main entry point: parse args, run classification.
    """
    args = parse_arguments()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.out_file)
    log_file = Path(args.log_file)

    # Run classification
    classify_by_reaction_core(input_csv, output_csv, log_file)

if __name__ == "__main__":
    main()