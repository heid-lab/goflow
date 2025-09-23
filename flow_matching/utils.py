import torch
from torch import Tensor
import numpy as np

from ase import Atoms

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from torch_geometric.data import Data

from flow_matching.chirality_forces import get_pseudoforces_batch
from scipy.spatial.distance import cdist

from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd

def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = True,
):
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(
            mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


def match_and_compute_rmsd(data, rtsp_i=1):
    mol_pred = Molecule(
        species=data.atom_type.long().cpu().numpy(),
        coords=data.pos_gen[-1].cpu().numpy(),
    )
    mol_ref = Molecule(
        species=data.atom_type.long().cpu().numpy(),
        coords=data.pos[:,1,:].cpu().numpy(),
    )
    try:
        rmsd = pymatgen_rmsd(
            mol_pred,
            mol_ref,
            ignore_chirality=True,
            threshold=0.5,
            same_order=False,
        )
    except:
        pred_pos_N_3, gt_pos_N_3 = pred_atom_index_align(data.smiles, data.pos[:, rtsp_i, :], data.pos_gen[-1])
        rmsd = rmsd_loss(pred_pos_N_3, gt_pos_N_3)

    return rmsd


def compute_steric_clash_penalty(
    coords_N_3: torch.Tensor, r_threshold: float = 0.7, epsilon: float = 1.0
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

def rmsd_median_loss(pred_N_3, gt_N_3):
    return torch.median(torch.linalg.vector_norm(pred_N_3 - gt_N_3, dim=1))

def rmsd_loss(pred_N_3: Tensor, gt_N_3: Tensor) -> Tensor:
    return torch.sqrt(torch.mean((pred_N_3 - gt_N_3) ** 2))

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


def get_shortest_path_x_1_(x_0_N_3, x_1_N_3):
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


@torch.no_grad()
def get_shortest_path_x_1(
    x_target_N_3: torch.Tensor,  # e.g., x0: (N,3) — fixed target
    x_moving_N_3: torch.Tensor,  # e.g., x1: (N,3) — will be rotated/translated
    return_aligned: bool = True
):
    """
    Kabsch alignment: find R,t that minimize || (x_moving R + t) - x_target ||_F.
    Returns R (3x3), t (3,), rmsd (scalar), and optionally the aligned coords.
    """
    assert x_moving_N_3.shape == x_target_N_3.shape and x_moving_N_3.shape[-1] == 3

    # 1) center both point sets
    c_moving_1_3 = x_moving_N_3.mean(dim=0, keepdim=True)  # (1,3)
    c_target_1_3 = x_target_N_3.mean(dim=0, keepdim=True)  # (1,3)
    X = x_moving_N_3 - c_moving_1_3                          # (N,3)
    Y = x_target_N_3 - c_target_1_3                          # (N,3)

    # 2) covariance and SVD
    # Move "moving" onto "target": M = X^T Y
    M_3_3 = X.transpose(0, 1) @ Y                            # (3,3)
    U, S, Vt = torch.linalg.svd(M_3_3, full_matrices=False)  # U (3,3), Vt (3,3)

    # 3) rotation (proper, no reflection)
    V = Vt.transpose(0, 1)
    R_3_3 = V @ U.transpose(0, 1)                            # (3,3)
    if torch.det(R_3_3) < 0:
        # flip last column of V (== last row of Vt) and recompute
        V[:, -1] *= -1
        R_3_3 = V @ U.transpose(0, 1)

    # 4) translation so that (x_moving R + t) best matches x_target
    # Using row-vector convention: x' = x R + t
    t_3 = (c_target_1_3 - c_moving_1_3 @ R_3_3).squeeze(0)   # (3,)

    # 5) rmsd
    if return_aligned:
        x_aligned_N_3 = x_moving_N_3 @ R_3_3 + t_3           # (N,3)
        rmsd = torch.sqrt(torch.mean((x_aligned_N_3 - x_target_N_3) ** 2))
        #return R_3_3, t_3, rmsd, x_aligned_N_3
        return x_aligned_N_3
    else:
        # Equivalent RMSD via centered coords
        X_rot = X @ R_3_3
        rmsd = torch.sqrt(torch.mean((X_rot - Y) ** 2))
        return R_3_3, t_3, rmsd
        

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

    """
    #tags inverted
    new_xyz = m.GetConformer().GetPositions() * -1
    for i in range(m.GetNumAtoms()):
        x,y,z = new_xyz[i]
        m.GetConformer().SetAtomPosition(i,Point3D(x,y,z))
    Chem.AssignStereochemistryFrom3D(m)
    pred_inv_chiral_tags = [a.GetChiralTag() for a in m.GetAtoms()]
    """
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


def gen_rdkit_2dconformer(smile):
    mol = Chem.MolFromSmarts(smile)
    mol.Compute2DCoords()
    pos = mol.GetConformer().GetPositions()
    return pos


def gen_rdkit_conformer(smile):
    mol = Chem.MolFromSmarts(smile)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    pos = mol.GetConformer().getPositions()
    return pos
