import pandas as pd
import numpy as np
import pickle
import argparse
import ast
import random
from pathlib import Path
from collections import defaultdict

def rxn_random_split(args):
    """
    Create a pickle file with a dict containing random rxn indices for train,val,test set.
    80% train, 10% val, 10% test
    """
    # Read the input CSV file
    df = pd.read_csv(args.input_rxn_csv)
    
    # Get all reaction indices
    all_indices = df['rxn'].tolist()
    
    # Shuffle the indices
    np.random.shuffle(all_indices)
    
    # Calculate split sizes
    total_size = len(all_indices)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    # Split indices
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    # Create dictionary with splits
    split_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    test_no_data_leakage(split_dict, set(all_indices))
    
    # Save to pickle file
    with open(Path(args.output_rxn_indices_path) / Path('random_split.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)
    
    print(f"Random split created: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


def rxn_core_split(args):
    """
    Create a pickle file with a dict containing the reaction_indices for train,val,test set.
    Assign the rxn core (type) clusters randomly to either train or [val, test] (cluster members can overlap in val&test)
    If assigned rxns not approximately 80%, 20% (+-2%), then do the random assignment again until sizes correct
    """
    # Read the input CSV file
    rxn_df = pd.read_csv(args.input_rxn_csv)
    
    # Read the reaction core clusters CSV
    clusters_df = pd.read_csv(args.rxn_core_clusters_csv)
    
    # Get all possible reaction indices
    all_indices = set(rxn_df['rxn'].tolist())
    
    # Process clusters to get reaction indices for each cluster
    clusters = []
    for _, row in clusters_df.iterrows():
        # Parse the reaction indices from string format "[1, 2, 3]" to actual list
        rxn_indices = ast.literal_eval(row['reaction_indices'])
        # Only include indices that are in the input reactions
        rxn_indices = [idx for idx in rxn_indices if idx in all_indices]
        if rxn_indices:  # Only add if there are valid indices
            clusters.append(rxn_indices)
    
    # Function to check if split is within acceptable range (80/20 ±2%)
    def is_valid_split(train_size, total_size):
        train_ratio = train_size / total_size
        return 0.78 <= train_ratio <= 0.82
    
    # Try splitting until we get an acceptable distribution
    valid_split = False
    max_attempts = 100
    attempts = 0
    
    while not valid_split and attempts < max_attempts:
        attempts += 1
        
        # Randomly assign clusters to train or val/test
        train_indices = []
        valtest_indices = []
        
        for cluster in clusters:
            if random.random() < 0.8:  # 80% chance to assign to train
                train_indices.extend(cluster)
            else:
                valtest_indices.extend(cluster)
        
        # Convert to sets to remove duplicates
        train_indices = list(set(train_indices))
        valtest_indices = list(set(valtest_indices))
        
        # Check for overlaps (reactions that appear in both sets)
        train_set = set(train_indices)
        valtest_set = set(valtest_indices)
        overlap = train_set.intersection(valtest_set)
        
        # Remove overlaps
        if overlap:
            print("Training and test sets overlapping for some reason. Trying removal.")
            for idx in overlap:
                if random.random() < 0.5:
                    train_indices.remove(idx)
                else:
                    valtest_indices.remove(idx)
        
        # Check if distribution is acceptable
        total_size = len(train_indices) + len(valtest_indices)
        train_size = len(train_indices)
        
        valid_split = is_valid_split(train_size, total_size)
    
    if not valid_split:
        print(f"Warning: Could not achieve desired split after {max_attempts} attempts")
    
    # Randomly split valtest into val and test
    np.random.shuffle(valtest_indices)
    val_size = len(valtest_indices) // 2
    val_indices = valtest_indices[:val_size]
    test_indices = valtest_indices[val_size:]
    
    # Create dictionary with splits
    split_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    test_no_data_leakage(split_dict, set(all_indices))
    
    # Save to pickle file
    with open(Path(args.output_rxn_indices_path) / Path('rxn_core_split.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)
    
    print(f"Core split created: train={len(train_indices)} ({len(train_indices)/total_size:.2f}), "
          f"val={len(val_indices)}, test={len(test_indices)}")


def rxn_barrier_split(args):
    """
    Create a pickle file with a dict containing rxn indices for train,val,test set.
    From the top 10%, and bottom 10% of EAs, randomly add 10% to validation, 10% to test (top and bottom EAs mixed)
    """
    # Read the input CSV file
    df = pd.read_csv(args.input_rxn_csv)
    
    # Sort reactions by energy barrier (dE0)
    df_sorted = df.sort_values(by='dE0')
    
    # Calculate the number of reactions for each 10% chunk
    total_rxns = len(df_sorted)
    chunk_size = total_rxns // 10
    
    # Get the bottom 10% and top 10% of barrier heights
    bottom_10_pct = df_sorted.iloc[:chunk_size]['rxn'].tolist()
    top_10_pct = df_sorted.iloc[-chunk_size:]['rxn'].tolist()
    middle_80_pct = df_sorted.iloc[chunk_size:-chunk_size]['rxn'].tolist()
    
    # Combine top and bottom 10%
    extreme_barriers = bottom_10_pct + top_10_pct
    
    # Shuffle the extreme barriers
    np.random.shuffle(extreme_barriers)
    
    # Split the extreme barriers into validation and test sets
    mid_point = len(extreme_barriers) // 2
    val_indices = extreme_barriers[:mid_point]
    test_indices = extreme_barriers[mid_point:]
    
    # Use the middle 80% for training
    train_indices = middle_80_pct
    
    # Create dictionary with splits
    split_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    test_no_data_leakage(split_dict, set(df['rxn'].tolist()))
    test_barrier_split_integrity(split_dict, df)
    
    # Save to pickle file
    with open(Path(args.output_rxn_indices_path) / Path('barrier_split.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)
    
    print(f"Barrier split created: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


def test_no_data_leakage(split_dict, all_indices):
    """
    Test that there is no data leakage between train, validation, and test sets.
    
    Args:
        split_dict: Dictionary with 'train', 'val', 'test' keys containing indices
        all_indices: Set of all available indices from the original dataset
    """
    train_set = set(split_dict['train'])
    val_set = set(split_dict['val'])
    test_set = set(split_dict['test'])
    
    # Test 1: No overlap between sets
    assert len(train_set.intersection(val_set)) == 0, "Overlap found between train and validation sets"
    assert len(train_set.intersection(test_set)) == 0, "Overlap found between train and test sets"
    assert len(val_set.intersection(test_set)) == 0, "Overlap found between validation and test sets"
    
    # Test 2: All indices are used (no missing data)
    all_used_indices = train_set.union(val_set).union(test_set)
    assert len(all_used_indices) == len(all_indices), f"Some indices are missing: expected {len(all_indices)}, got {len(all_used_indices)}"
    assert all_used_indices == set(all_indices), "The sets of indices don't match"
    
    # Test 3: Each index appears exactly once
    counts = defaultdict(int)
    for idx in split_dict['train'] + split_dict['val'] + split_dict['test']:
        counts[idx] += 1
    
    assert all(count == 1 for count in counts.values()), "Some indices appear multiple times across sets"
    
    print("All data leakage tests passed!")

    
def test_barrier_split_integrity(split_dict, df):
    """
    Test that the barrier split correctly separates the data by energy barriers
    """
    # Get energy barriers for each set
    train_barriers = df[df['rxn'].isin(split_dict['train'])]['dE0'].values
    val_test_barriers = df[df['rxn'].isin(split_dict['val'] + split_dict['test'])]['dE0'].values
    
    # Check that train contains the middle values
    min_train = np.min(train_barriers)
    max_train = np.max(train_barriers)
    
    # Count how many val/test barriers are outside the train range
    extremes_count = np.sum((val_test_barriers < min_train) | (val_test_barriers > max_train))
    
    # At least 80% of val/test should be outside train range for proper barrier split
    assert extremes_count >= 0.8 * len(val_test_barriers), "Barrier split doesn't properly separate by energy barriers"
    
    print("Barrier split integrity verified!")

    
if __name__ == "__main__":
    """
    Example of input_rxn_csv:
    rxn,smiles,dE0
    0,[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]>>[C:1]([C:2]([N:3]=[O:4])=[N+:6]=[N-:5])([H:7])([H:8])[H:9],48.61085
    ...
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_rxn_csv", type=str, help="path to file containing the rxn id,smiles,ea")
    parser.add_argument("--output_rxn_indices_path", type=str, help="path to pickle file containing the created rxn indices for train,val,test")
    
    # Random split ---------------
    parser.add_argument("--random", action="store_true", default=False)
    
    # Rxn core split --------------
    parser.add_argument("--rxn_core_clusters", action="store_true", default=False)
    parser.add_argument("--rxn_core_clusters_csv", type=str, help="path to csv with reaction type clusters")
    
    # Ea split ---------------
    parser.add_argument("--barrier_height", action="store_true", default=False)
    args = parser.parse_args()
    
    # Call the appropriate function based on arguments
    if args.random:
        rxn_random_split(args)
    
    if args.rxn_core_clusters:
        if not args.rxn_core_clusters_csv:
            raise ValueError("rxn_core_clusters_csv must be specified for reaction core split")
        rxn_core_split(args)
    
    if args.barrier_height:
        rxn_barrier_split(args)
    else:
        print("No split method specified. Please use --random, --rxn_core_clusters, or --barrier_height")
