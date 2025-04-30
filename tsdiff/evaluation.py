import torch
import numpy as np
from ase.calculators.lj import LennardJones
from ase import Atoms
import math
from pathlib import Path
from typing import Any, Dict
import pickle
from torch import Tensor
from torch_geometric.data import Data
import rdkit.Chem as Chem
from scipy.spatial.distance import cdist
import pandas as pd
from collections import defaultdict
import itertools
import argparse
import os


def calculate_pot_energy(mol: Atoms):
    mol.calc = LennardJones()
    return mol.get_potential_energy()


def compute_steric_clash_penalty(
    coords_N_3: torch.Tensor, r_threshold: float = 1.2, epsilon: float = 1.0
) -> torch.Tensor:
    """
    Compute a steric clash penalty based on a simplified LJ
    potential which only includes the repulsive 12-term. For any pair of atoms,
    if the distance r satisfies r < r_threshold then we add a penalty:

        V(r) = epsilon * [(r_threshold / r)^{12} - 1]    for r < r_threshold
        V(r) = 0                                          for r >= r_threshold

    The default r_threshold of 1.2 Å is chosen based on the observation that the
    shortest possible bond (C-C triple bond) is around this length.

    Parameters:
        coords_N_3 (torch.Tensor): Tensor of shape (N,3) with 3D positions.
        r_threshold (float): Distance threshold below which a steric clash is 
                             penalized.
        epsilon (float): Scaling factor for the penalty.

    Returns:
        torch.Tensor: The total steric clash penalty for the entire set of atoms.
    """
    # Compute all pairwise distances
    dists_N_N = torch.cdist(coords_N_3, coords_N_3, p=2)
    # Consider only unique pairs (i < j)
    mask = torch.triu(torch.ones_like(dists_N_N, dtype=torch.bool), diagonal=1)
    dists_K = dists_N_N[mask]  # K = N*(N-1)/2

    # Identify clashes where the distance is below the threshold.
    clash_mask = dists_K < r_threshold
    penalty_K = torch.zeros_like(dists_K)
    
    penalty_K[clash_mask] = epsilon * ((r_threshold / dists_K[clash_mask]) ** 12 - 1)
    total_penalty = penalty_K.sum()
    return total_penalty


def create_ase_molecule(atomic_numbers: np.ndarray, positions: np.ndarray):
    if positions.shape[0] != len(atomic_numbers):
        raise ValueError(f"Number of positions ({positions.shape[0]}) does not match the number of atoms ({len(atomic_numbers)}).")

    return Atoms(numbers=atomic_numbers, positions=positions)

def rmsd_loss(pred_N_3, gt_N_3):
    return torch.linalg.vector_norm(pred_N_3 - gt_N_3, dim=1).mean()

def rmsd_median_loss(pred_N_3, gt_N_3):
    return torch.median(torch.linalg.vector_norm(pred_N_3 - gt_N_3, dim=1))

def get_shortest_path_fast_batched_x_1(x_0_N_3, x_1_N_3, batch):
    # x_0_N_3, x_1_N_3 are tensors of shape (N, 3)
    # batch.batch is a 1D tensor of length N with group indices.
    device = x_0_N_3.device
    Nm = int(batch.batch.max().item() + 1)  # number of molecules in the batch

    # Compute the number of points per group.
    counts = torch.bincount(batch.batch, minlength=Nm).to(x_0_N_3.dtype)

    # Compute group centroids via index_add.
    centers_x0_Nm_3 = torch.zeros((Nm, 3), device=device)
    centers_x1_Nm_3 = torch.zeros((Nm, 3), device=device)
    centers_x0_Nm_3 = centers_x0_Nm_3.index_add(0, batch.batch, x_0_N_3)
    centers_x1_Nm_3 = centers_x1_Nm_3.index_add(0, batch.batch, x_1_N_3)
    centers_x0_Nm_3 = centers_x0_Nm_3 / counts.unsqueeze(1)
    centers_x1_Nm_3 = centers_x1_Nm_3 / counts.unsqueeze(1)

    # Center the points.
    x0_centered_N_3 = x_0_N_3 - centers_x0_Nm_3[batch.batch]
    x1_centered_N_3 = x_1_N_3 - centers_x1_Nm_3[batch.batch]

    # For each point, compute the outer product: shape (N, 3, 3)
    prod_N_3_3 = x1_centered_N_3.unsqueeze(2) * x0_centered_N_3.unsqueeze(1)
    # Sum the outer products per group to form (B, 3, 3) covariance matrices.
    M_Nm_3_3 = torch.zeros((Nm, 3, 3), device=device)
    M_Nm_3_3 = M_Nm_3_3.index_add(0, batch.batch, prod_N_3_3)

    # Compute the batched SVD
    U_Nm_3_3, S_Nm_3, Vt_Nm_3_3 = torch.linalg.svd(M_Nm_3_3)

    # Reflection correction per group.
    det_Nm = torch.det(torch.bmm(U_Nm_3_3, Vt_Nm_3_3))
    # (3, 3) -> (1, 3, 3) -> (Nm, 3, 3). That is, repeat the (3,3)-identity matrix Nm times.
    D_Nm_3_3 = torch.eye(3, device=device).unsqueeze(0).repeat(Nm, 1, 1)
    # Change the 2,2 element of the identity matrix to -1 if det < 0.
    D_Nm_3_3[det_Nm < 0, 2, 2] = -1
    # Apply the reflection correction.
    R_opt_Nm_3_3 = torch.bmm(U_Nm_3_3, torch.bmm(D_Nm_3_3, Vt_Nm_3_3))

    # Apply the optimal rotation:
    # For each point i (belonging to group j), we set:
    #   x1_aligned[i] = (x1[i] - centers_x1[j]) @ R_opt[j] + centers_x0[j]
    x_1_rotated_N_3 = torch.bmm(
        x1_centered_N_3.unsqueeze(1), # (N,3) -> (N,1,3)
        R_opt_Nm_3_3[batch.batch]   # (Nm,3,3) -> (N,3,3)
    ).squeeze(1) # (N,1,3) -> (N,3)
    x1_aligned_N_3 = x_1_rotated_N_3 + centers_x0_Nm_3[batch.batch]

    return x1_aligned_N_3


def get_shortest_path_batched_x_1(x_0_N_3, x_1_N_3, batch):
    x1_aligned_N_3 = torch.zeros_like(x_1_N_3)
    for j, data in enumerate(batch.to_data_list()):
        # Nm ... number of nodes in the molecule as opposed to N in the whole batch
        # Grab only the batch point clouds
        batch_mask = (batch.batch == j)
        x_0_Nm_3 = x_0_N_3[batch_mask]
        x_1_Nm_3 = x_1_N_3[batch_mask]

        # Center the points
        x0_center_3 = x_0_Nm_3.mean(dim=0, keepdim=True)
        x1_center_3 = x_1_Nm_3.mean(dim=0, keepdim=True)
        x_0_centered_Nm_3 = x_0_Nm_3 - x0_center_3
        x_1_centered_Nm_3 = x_1_Nm_3 - x1_center_3

        # Compute cross-covariance matrix
        M_3_3 = x_1_centered_Nm_3.t() @ x_0_centered_Nm_3
        U, _, Vt = torch.linalg.svd(M_3_3)

        # correct for reflection
        if torch.det(U @ Vt) < 0:
            U[:, -1] *= -1
        R_opt = U @ Vt

        x1_centered_rotated_Nm_3 = x_1_centered_Nm_3 @ R_opt
        # Add back the centroid of x_0
        x1_aligned_Nm_3 = x1_centered_rotated_Nm_3 + x0_center_3
        x1_aligned_N_3[batch_mask] = x1_aligned_Nm_3
    return x1_aligned_N_3


def get_shortest_path_x_1(x_0_N_3, x_1_N_3):
    """
    Find the rotation matrix R such that the distance of the (Nx3) matrix of ground truth(GT) atomic positions x_1_N_3 rotated with R
    to the random atomic positions x_0_N_3 is minimized. More formally:

    R_opt = argmin_R(dist(x_0_N_3, x_1_N_3 @ R.T))
    x_1_opt_N_3 = x_1_N_3 @ R_opt.T
    return x_1_opt_N_3

    This is used so the model learns to flow (diffuse) towards the GT atomic positions, but with the shortest distance.
    Useful because it could flow towards any rotated version of the GT, which are all valid. But we want a unique, shortest path.
    """
    # Center the points
    x0_center = x_0_N_3.mean(dim=0, keepdim=True)
    x1_center = x_1_N_3.mean(dim=0, keepdim=True)
    x_0_centered = x_0_N_3 - x0_center
    x_1_centered = x_1_N_3 - x1_center

    # Compute cross-covariance matrix
    M_3_3 = x_1_centered.t() @ x_0_centered

    U, _, Vt = torch.linalg.svd(M_3_3)

    # correct for reflection
    if torch.det(U @ Vt) < 0:
        U[:, -1] *= -1

    R_opt = U @ Vt
    x1_centered_rotated = x_1_centered @ R_opt
    # Add back the centroid of x_0
    x1_aligned = x1_centered_rotated + x0_center
    return x1_aligned


def get_min_rmsd_match(matches, gt_pos, pred_pos):
    rmsd_M = []
    for match in matches:
        pred_pos_match = pred_pos[list(match)]
        gt_pos_aligned = get_shortest_path_x_1(pred_pos_match, gt_pos)
        rmsd_M.append(rmsd_loss(pred_pos_match, gt_pos_aligned))
    return list(matches[rmsd_M.index(min(rmsd_M))])


def calc_DMAE(dm_ref, dm_guess, mape=False):
    if mape:
        retval = abs(dm_ref - dm_guess) / dm_ref
    else:
        retval = abs(dm_ref - dm_guess)
    return np.triu(retval, k=1).sum() / len(dm_ref) / (len(dm_ref) - 1) * 2


def get_min_dmae_match(matches, ref_pos, prb_pos):
    dmaes = []
    for match in matches:
        match_pos = prb_pos[list(match)]
        dmae = calc_DMAE(cdist(ref_pos, ref_pos), cdist(match_pos, match_pos))
        dmaes.append(dmae)
    return list(matches[dmaes.index(min(dmaes))])


def get_substruct_matches(smarts):
    smarts_r, smarts_p = smarts.split(">>")
    mol_r = Chem.MolFromSmarts(smarts_r)
    mol_p = Chem.MolFromSmarts(smarts_p)

    matches_r = list(mol_r.GetSubstructMatches(mol_r, uniquify=False, useChirality=True))
    map_r = np.array([atom.GetAtomMapNum() for atom in mol_r.GetAtoms()]) - 1
    map_r_inv = np.argsort(map_r)
    for i in range(len(matches_r)):
        matches_r[i] = tuple(map_r[np.array(matches_r[i])[map_r_inv]])

    matches_p = list(mol_p.GetSubstructMatches(mol_p, uniquify=False, useChirality=True))
    map_p = np.array([atom.GetAtomMapNum() for atom in mol_p.GetAtoms()]) - 1
    map_p_inv = np.argsort(map_p)
    for i in range(len(matches_p)):
        matches_p[i] = tuple(map_p[np.array(matches_p[i])[map_p_inv]])

    matches = set(matches_r) & set(matches_p)
    matches = list(matches)
    matches.sort()
    return matches


def pred_atom_index_align(smiles, gt_atom_pos, pred_atom_pos):
    matches = get_substruct_matches(smiles)
    match = get_min_rmsd_match(matches, gt_atom_pos, pred_atom_pos)

    pred_atom_pos_match = pred_atom_pos[match]
    gt_atom_pos_aligned = get_shortest_path_x_1(pred_atom_pos_match, gt_atom_pos)

    return pred_atom_pos_match, gt_atom_pos_aligned


def pred_atom_index_align_mad(smiles, gt_atom_pos, pred_atom_pos) -> Tensor:
    matches = get_substruct_matches(smiles)
    match = get_min_dmae_match(matches, gt_atom_pos, pred_atom_pos)
    return pred_atom_pos[match]

# --------------------------- Metric Code Start ---------------------------

def build_connectivity(edge_index: torch.Tensor) -> dict:
    """
    Build a connectivity dictionary from an edge_index tensor.
    
    Parameters:
        edge_index (torch.Tensor): Tensor of shape (2, E) representing bond connections.
    
    Returns:
        dict: A dictionary mapping each atom index to a sorted list of its neighbors.
    """
    connectivity = defaultdict(set)
    
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    
    for i, j in zip(src, dst):
        connectivity[i].add(j)
        connectivity[j].add(i)
    
    connectivity = {node: sorted(neigh) for node, neigh in connectivity.items()}
    return connectivity


def extract_angles_from_connectivity(connectivity: dict) -> torch.Tensor:
    """
    Extract angle (triplet) indices given the connectivity dictionary.
    
    For each central atom j that has at least two neighbors,
    form (i, j, k) for every unique pair {i, k}.

    Parameters:
        connectivity (dict): Dictionary mapping node -> list of neighbors.
    
    Returns:
        torch.Tensor: Tensor of shape (num_angles, 3) where each row is (i, j, k) with j as the vertex.
    """
    angles = []
    for j, neighbors in connectivity.items():
        if len(neighbors) < 2:
            continue
        # Use combinations to generate all unique pairs
        for i, k in itertools.combinations(neighbors, 2):
            angles.append([i, j, k])
    if angles:
        return torch.tensor(angles, dtype=torch.long)
    else:
        return torch.empty((0, 3), dtype=torch.long)


def extract_dihedrals_from_connectivity(connectivity: dict) -> torch.Tensor:
    """
    Extract dihedral (quadruplet) indices from the connectivity dictionary.
    
    For every bond (j,k), for every neighbor i of j (excluding k)
    and every neighbor l of k (excluding j), form (i, j, k, l).

    Parameters:
        connectivity (dict): Dictionary mapping node -> list of neighbors.
    
    Returns:
        torch.Tensor: Tensor of shape (num_dihedrals, 4) with each row being (i, j, k, l).
    """
    dihedrals = []
    for j, neighbors_j in connectivity.items():
        for k in neighbors_j:
            # For bond (j, k), iterate over neighbors of j and k, excluding the counterpart.
            for i in neighbors_j:
                if i == k:
                    continue
                # Get neighbors of k; if none, skip.
                neighbors_k = connectivity.get(k, [])
                for l in neighbors_k:
                    if l == j:
                        continue
                    dihedrals.append([i, j, k, l])
    if dihedrals:
        return torch.tensor(dihedrals, dtype=torch.long)
    else:
        return torch.empty((0, 4), dtype=torch.long)


def compute_bond_angles(
    coords_N_3: torch.Tensor, angle_indices_M_3: torch.Tensor
) -> torch.Tensor:
    """
    Compute the bond angles for a set of triplets. Each triplet is assumed
    to be (i, j, k) with j as the vertex. The bond angle is computed as

        θ = arccos [ (vec1 · vec2) / (||vec1|| ||vec2||) ]

    Parameters:
        coords_N_3 (torch.Tensor): Tensor of shape (N,3) with 3D positions.
        angle_indices_M_3 (torch.Tensor): Tensor of shape (M,3) with atom indices
                                      defining each angle = (i, j, k).

    Returns:
        torch.Tensor: A tensor of shape (M,) with the angles in radians.
    """
    vec1_M_3 = coords_N_3[angle_indices_M_3[:, 0]] - coords_N_3[angle_indices_M_3[:, 1]]
    vec2_M_3 = coords_N_3[angle_indices_M_3[:, 2]] - coords_N_3[angle_indices_M_3[:, 1]]
    dot_prod_M = (vec1_M_3 * vec2_M_3).sum(dim=1)
    norm1_M = vec1_M_3.norm(dim=1)
    norm2_M = vec2_M_3.norm(dim=1)
    cosine_M = dot_prod_M / (norm1_M * norm2_M + 1e-9)
    # Clamp to [-1,1] to avoid numerical issues with arccos
    cosine_M = torch.clamp(cosine_M, -1.0, 1.0)
    angles_M = torch.acos(cosine_M)
    return angles_M


def compute_dihedral_angles(
    coords_N_3: torch.Tensor, dihedral_indices_M_4: torch.Tensor
) -> torch.Tensor:
    """
    Compute the dihedral (torsion) angles for a set of quadruplets.
    A dihedral is defined by four atoms with indices (i, j, k, l).

    Parameters:
        coords_N_3 (torch.Tensor): Tensor of shape (N,3) with 3D positions.
        dihedral_indices_M_4 (torch.Tensor): Tensor of shape (M,4) with atom indices
                                           defining each dihedral.

    Returns:
        torch.Tensor: A tensor of shape (M,) with the dihedral angles in radians.
    """
    p0_M_3 = coords_N_3[dihedral_indices_M_4[:, 0]]
    p1_M_3 = coords_N_3[dihedral_indices_M_4[:, 1]]
    p2_M_3 = coords_N_3[dihedral_indices_M_4[:, 2]]
    p3_M_3 = coords_N_3[dihedral_indices_M_4[:, 3]]

    b0_M_3 = p1_M_3 - p0_M_3
    b1_M_3 = p2_M_3 - p1_M_3
    b2_M_3 = p3_M_3 - p2_M_3

    n1_M_3 = torch.cross(b0_M_3, b1_M_3, dim=1)
    n2_M_3 = torch.cross(b1_M_3, b2_M_3, dim=1)

    n1_norm_M_3 = n1_M_3 / (n1_M_3.norm(dim=1, keepdim=True) + 1e-9)
    n2_norm_M_3 = n2_M_3 / (n2_M_3.norm(dim=1, keepdim=True) + 1e-9)
    b1_unit_M_3 = b1_M_3 / (b1_M_3.norm(dim=1, keepdim=True) + 1e-9)

    m1_M_3 = torch.cross(n1_norm_M_3, b1_unit_M_3, dim=1)

    x_M = (n1_norm_M_3 * n2_norm_M_3).sum(dim=1)
    y_M = (m1_M_3 * n2_norm_M_3).sum(dim=1)

    dihedral_angles_M = torch.atan2(y_M, x_M)
    return dihedral_angles_M


def evaluate_geometry(
    data: Data,
    r_threshold: float = 1.2,
    epsilon: float = 1.0,
) -> Dict[str, float]:
    """
    Parameters:
    data (torch_geometric.data.Data): Reaction data
    dihedral_indices (torch.Tensor): Tensor (M, 4) of indices for dihedral angles.
    r_threshold (float): Distance threshold for steric clash penalty.
    epsilon (float): Scaling factor for the clash penalty.

    Returns:
    Dict[str, float]
    """
    rtsp_i = 1
    # RMSE error
    pred_pos_N_3, gt_pos_N_3 = pred_atom_index_align(data.smiles, data.pos[:, rtsp_i, :], data.pos_gen)
    rmse = rmsd_loss(pred_pos_N_3, gt_pos_N_3)
    
    # MAE error
    pred_pos_aligned_mae = pred_atom_index_align_mad(data.smiles, data.pos[:, rtsp_i, :], data.pos_gen)
    mae = calc_DMAE(cdist(gt_pos_N_3, gt_pos_N_3), cdist(pred_pos_aligned_mae, pred_pos_aligned_mae))
    
    connectivity = build_connectivity(data.edge_index)
    # Extract angle and dihedral indices from the connectivity.
    angle_indices_M1_3 = extract_angles_from_connectivity(connectivity)
    dihedral_indices_M2_4 = extract_dihedrals_from_connectivity(connectivity)

    # Bond angle comparison.
    gt_angles_M1 = compute_bond_angles(gt_pos_N_3, angle_indices_M1_3)
    pred_angles_M1 = compute_bond_angles(pred_pos_N_3, angle_indices_M1_3)
    # Convert radians to degrees.
    bond_angle_error = (torch.abs(gt_angles_M1 - pred_angles_M1) * 180.0 / math.pi).mean()

    # Dihedral angle comparison.
    gt_dihedrals_M2 = compute_dihedral_angles(gt_pos_N_3, dihedral_indices_M2_4)
    pred_dihedrals_M2 = compute_dihedral_angles(pred_pos_N_3, dihedral_indices_M2_4)
    diff_M2 = torch.abs(gt_dihedrals_M2 - pred_dihedrals_M2)
    # Handle periodicity: if the difference is larger than pi, wrap around.
    diff_M2 = torch.where(diff_M2 > math.pi, 2 * math.pi - diff_M2, diff_M2)
    dihedral_angle_error = (diff_M2 * 180.0 / math.pi).mean()

    # Steric clash penalty
    steric_clash_pred = compute_steric_clash_penalty(pred_pos_N_3, r_threshold, epsilon)
    steric_clash_gt = compute_steric_clash_penalty(gt_pos_N_3, r_threshold, epsilon)
    steric_clash_diff = (steric_clash_pred - steric_clash_gt).item()
    steric_clash_diff = min(steric_clash_diff, 9999)

    return {
        "mae": float(mae),
        "rmse": rmse.item(),
        "angle_error": bond_angle_error.item(),
        "dihedral_error": dihedral_angle_error.item(),
        "steric_clash": steric_clash_diff,
    }

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate geometry prediction results')
    parser.add_argument('--pred_data', type=str, required=True, 
                        help='Path to the prediction data file (.pkl)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=1.2,
                        help='Distance threshold for steric clash penalty (default: 1.2)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Scaling factor for clash penalty (default: 1.0)')
    parser.add_argument("--split_name", type=str, required=True, help="name of the split")
    
    args = parser.parse_args()
    
    # Ensure save directory exists
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load prediction data
    print(f"Loading prediction data from {args.pred_data}")
    with open(args.pred_data, "rb") as f:
        data_R = pickle.load(f)
    
    print(f"Processing {len(data_R)} samples...")
    pd_results_R = [evaluate_geometry(data, 
                                     r_threshold=args.threshold, 
                                     epsilon=args.epsilon) for data in data_R]
    
    # Calculate and save statistics
    pd_results_df = pd.DataFrame(pd_results_R)
    means = pd_results_df.mean()
    
    run_id = f"{args.split_name}"
    
    # Check if stats file exists and load it, otherwise create new
    stats_path = save_path / 'stats.csv'
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path, index_col=0)
        # Add new row with current results
        stats_df.loc[run_id] = means
    else:
        # Create new dataframe with just the mean
        stats_df = pd.DataFrame([means], index=[run_id])
    
    # Save updated stats
    stats_df.to_csv(stats_path, float_format='%.3f')
    print(f"Statistical summary appended to {stats_path}")
    
    # Print summary (mean only)
    print("\nEvaluation Summary:")
    for metric in ['mae', 'rmse', 'angle_error', 'dihedral_error', 'steric_clash']:
        print(f"{metric:15s}: {means[metric]:.3f}")
