import torch
from torch import Tensor
import numpy as np

from ase import Atoms

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from scipy.spatial.distance import cdist


def compute_steric_clash_penalty(
    coords_N_3: torch.Tensor, r_threshold: float = 0.7, epsilon: float = 1.0
) -> torch.Tensor:
    """
    Compute a steric clash penalty based on a simplified LJ
    potential which only includes the repulsive 12-term. For any pair of atoms,
    if the distance r satisfies r < r_threshold then we add a penalty:

        V(r) = epsilon * [(r_threshold / r)^12 - 1]    for r < r_threshold
        V(r) = 0                                          for r >= r_threshold

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

def remove_all_bonds(mol):
    rw_mol = Chem.RWMol(mol)
    # Collect bonds to avoid modifying the list while iterating
    bonds_to_remove = []
    for bond in rw_mol.GetBonds():
        bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    for i, j in bonds_to_remove:
        rw_mol.RemoveBond(i, j)
    return rw_mol.GetMol()

def assign_positions_from_atom_map(smiles, positions):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    
    mol = remove_all_bonds(mol)

    conf = Chem.Conformer(mol.GetNumAtoms())
    
    atom_mapping = [(atom.GetAtomMapNum(), idx) for idx, atom in enumerate(mol.GetAtoms())]
    atom_mapping.sort(key=lambda tup: tup[0])
    
    # Assign positions according to the sorted order (assumes positions is ordered by map number)
    for pos, (_, atom_idx) in zip(positions, atom_mapping):
        conf.SetAtomPosition(atom_idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
        
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    
    return mol


def is_correct_chirality(smiles: str, pos_gt_N_3: Tensor, pos_pred_N_3: Tensor):
    #tags ground truth
    m = assign_positions_from_atom_map(smiles.split('>>')[0], pos_gt_N_3)
    rdDetermineBonds.DetermineConnectivity(m)
    Chem.AssignStereochemistryFrom3D(m)
    gt_chiral_tags = [a.GetChiralTag() for a in m.GetAtoms()]

    #tags predicted
    m = assign_positions_from_atom_map(smiles.split('>>')[0], pos_pred_N_3)
    rdDetermineBonds.DetermineConnectivity(m)
    Chem.AssignStereochemistryFrom3D(m)
    pred_chiral_tags = [a.GetChiralTag() for a in m.GetAtoms()]

    return not (False in [t1 == t2 for t1, t2 in zip(gt_chiral_tags, pred_chiral_tags)])


def no_steric_clash(pos_N_3: Tensor, clash_thresh=25):
    return compute_steric_clash_penalty(pos_N_3, r_threshold=0.7) < clash_thresh


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
    # match = get_min_dmae_match(matches, gt_atom_pos, pred_atom_pos)
    match = get_min_dmae_match(matches, gt_atom_pos, pred_atom_pos)
    return pred_atom_pos[match]


def calc_DMAE_torch(dm_ref, dm_guess, mape=False):
    """
    Compute the Distance Matrix Absolute Error (DMAE) between two distance matrices.
    dm_ref and dm_guess are torch tensors of shape (N, N).
    """
    if mape:
        diff = torch.abs(dm_ref - dm_guess) / dm_ref
    else:
        diff = torch.abs(dm_ref - dm_guess)
    # Keep only the upper triangle (excluding the diagonal)
    diff_upper = torch.triu(diff, diagonal=1)
    N = dm_ref.shape[0]
    return 2 * diff_upper.sum() / (N * (N - 1))


def get_min_dmae_match_torch_batch(matches_M_N, pos_gt_N_3, pos_pred_S_N_3, mape=False):
    """
    Given a set of matches (each a tuple of indices), ground-truth positions (pos_gt_N_3), 
    and S samples of predicted positions (pos_pred_S_N_3), compute the DMAE for each match
    in each sample and return the match (as a list) with the minimal DMAE per sample.
    
    Args:
        matches_M_N: list or tensor of candidate matches of shape (M, N)
                     where M is the number of candidate matches and N is the number of atoms.
        pos_gt_N_3:  tensor of ground-truth atom positions of shape (N, 3).
        pos_pred_S_N_3: tensor of predicted atom positions for S samples (shape: S, N_total, 3)
                        where N_total must be large enough to index by matches_M_N.
        mape:        Boolean flag. If True, compute Mean Absolute Percentage Error instead of 
                     absolute differences.
    
    Returns:
        A list (or tensor) of best candidate match indices for each sample, of shape (S, N).  
        Each row corresponds to the candidate match (from matches_M_N) that minimizes the DMAE 
        for that sample.
    """
    # Ensure matches_M_N is a tensor on the same device as pos_pred_S_N_3.
    matches_M_N = torch.tensor(matches_M_N, dtype=torch.long, device=pos_pred_S_N_3.device)
    
    # Candidate predicted positions indexing:
    # For each sample (S) and each candidate match (M), select positions for N atoms.
    candidate_pred_pos_S_M_N_3 = pos_pred_S_N_3[:, matches_M_N]
    
    S, M, N, _ = candidate_pred_pos_S_M_N_3.shape
    
    # Flatten the first two dimensions (S and M) for batch distance computation.
    candidate_pred_pos_SM_N_3 = candidate_pred_pos_S_M_N_3.reshape(S * M, N, 3)
    
    # Compute pairwise distance matrices for each candidate match.
    d_matches_SM_N_N = torch.cdist(candidate_pred_pos_SM_N_3, candidate_pred_pos_SM_N_3, p=2)    
    d_matches_S_M_N_N = d_matches_SM_N_N.reshape(S, M, N, N)
    
    # Compute the reference distance matrix from pos_gt_N_3
    d_ref_N_N = torch.cdist(pos_gt_N_3.unsqueeze(0), pos_gt_N_3.unsqueeze(0), p=2).squeeze(0)
    d_ref_1_1_N_N = d_ref_N_N.unsqueeze(0).unsqueeze(0)
    
    # Compute the error difference.
    if mape:
        diff_S_M_N_N = torch.abs(d_ref_1_1_N_N - d_matches_S_M_N_N) / d_ref_1_1_N_N
    else:
        diff_S_M_N_N = torch.abs(d_ref_1_1_N_N - d_matches_S_M_N_N)
    
    # Zero out the lower triangle (and diagonal) of each (N, N) distance matrix.
    diff_upper_S_M_N_N = torch.triu(diff_S_M_N_N, diagonal=1)
    
    # Calculate DMAE for each candidate match in each sample.
    # Normalization factor: (N*(N-1)/2) with a factor of 2 gives division by (N*(N-1))
    dmaes_S_M = 2 * diff_upper_S_M_N_N.sum(dim=(-1, -2)) / (N * (N - 1))  # Shape: (S, M)
    
    # For each sample, identify the candidate match with the smallest DMAE.
    best_idx_S = torch.argmin(dmaes_S_M, dim=1)  # Shape: (S,)
    
    # Retrieve the best candidate match indices for each sample.
    best_matches_S_N = matches_M_N[best_idx_S]
    
    return best_matches_S_N