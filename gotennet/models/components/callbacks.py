import time
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

import lightning.pytorch as L
import torch
from lightning import Trainer
from torch_geometric.data import Data

from flow_matching.utils import rmsd_loss, compute_steric_clash_penalty, pred_atom_index_align, calc_DMAE, pred_atom_index_align_mad
from gotennet.utils import RankedLogger
from scipy.spatial.distance import cdist

import itertools
from collections import defaultdict


log = RankedLogger(__name__, rank_zero_only=True)


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
        # generate all unique pairs
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

        theta = arccos [ (vec1 · vec2) / (||vec1|| ||vec2||) ]

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
    r_threshold (float): Distance threshold for steric clash penalty.
    epsilon (float): Scaling factor for the clash penalty.

    Returns:
    Dict[str, float]
    """
    # RMSE error
    pred_pos_N_3, gt_pos_N_3 = pred_atom_index_align(data.smiles, data.pos[:, 1, :], data.pos_gen[-1])
    rmse = rmsd_loss(pred_pos_N_3, gt_pos_N_3)
    # MAE error
    pred_pos_aligned_mae = pred_atom_index_align_mad(data.smiles, data.pos[:, 1, :], data.pos_gen[-1])
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
        "mae": round(float(mae), 4),
        "rmse": round(rmse.item(), 4),
        "angle_error": round(bond_angle_error.item(), 4),
        "dihedral_error": round(dihedral_angle_error.item(), 4),
        "steric_clash": round(steric_clash_diff, 4)
    }

# --------------------------- Metric Code End ---------------------------


class TestAndSaveResultsAfterTrainingCallback(L.Callback):
    def __init__(self, save_path, runs_stats_path=None):#, mr_stats_path=None):
        self.save_path = Path(save_path)
        self.runs_stats_path = Path(runs_stats_path)
        self._test_start_time = None

    def save_test_predictions(self, module):
        pickle_save_path = self.save_path / 'test_samples/samples_all.pkl'
        pickle_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pickle_save_path, "wb") as f:
            pickle.dump(module.results_R, f)

    def save_stats_to_csv(self, pd_results_mean, module):
        mr_stats_file = self.runs_stats_path / 'stats.csv'
        
        pd_results_mean['num_steps'] = module.num_steps
        pd_results_mean['num_samples'] = module.num_samples
        
        if mr_stats_file.exists():
            df = pd.read_csv(mr_stats_file)
            df = pd.concat([df, pd.DataFrame([pd_results_mean])], ignore_index=True)
            df.to_csv(mr_stats_file, index=False, float_format='%.3f')
        else:
            pd.DataFrame([pd_results_mean]).to_csv(mr_stats_file, index=False, float_format='%.3f')

    def on_test_start(self, trainer: Trainer, module: L.LightningModule):
        module.results_R = []
        self._test_start_time = time.perf_counter()
    
    def on_test_end(self, trainer: Trainer, module):
        inference_time_per_rxn = (time.perf_counter() - self._test_start_time) / len(module.results_R)
        self._test_start_time = None
        
        pd_results_R = [evaluate_geometry(data) for data in module.results_R]
        pd_results_df = pd.DataFrame(pd_results_R)
        pd_results_df.to_csv(self.save_path / 'test_results.csv', index=False, float_format='%.3f')

        means = pd_results_df.mean()
        means['time'] = inference_time_per_rxn

        self.save_stats_to_csv(means, module)
        self.save_test_predictions(module)


class EMALossCallback(L.Callback):
    """
    Exponential Moving Average (EMA) Loss Callback.
    This callback calculates and logs the EMA of the validation loss.
    """

    def __init__(
            self,
            alpha: float = 0.99,
            soft_beta: float = 10,
            validation_loss_name: str = "val_loss",
            ema_log_name: str = "validation/ema_loss"
    ):
        """
        Initialize the EMALossCallback.

        Args:
            alpha (float): The decay factor for EMA calculation. Default is 0.99.
            soft_beta (float): The soft beta factor for loss capping. Default is 10.
            validation_loss_name (str): The name of the validation loss in the outputs. Default is "val_loss".
            ema_log_name (str): The name under which to log the EMA loss. Default is "validation/ema_loss".
        """
        super().__init__()
        self.alpha = alpha
        self.ema: Optional[torch.Tensor] = None
        self.num_batches: int = 0
        self.soft_beta = soft_beta
        self.total_loss: Optional[torch.Tensor] = None
        self.validation_loss_name = validation_loss_name
        self.ema_log_name = ema_log_name

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state dictionary.

        Args:
            state_dict (Dict[str, Any]): The state dictionary to load from.
        """
        if "ema_loss" in state_dict:
            log.info("EMA loss loaded")
            self.ema = state_dict["ema_loss"]
        else:
            log.info("EMA loss not found in checkpoint")
            self.ema = None

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state dictionary.

        Returns:
            Dict[str, Any]: The state dictionary containing the EMA loss.
        """
        return {"ema_loss": self.ema}

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Called when the validation epoch begins.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
        """
        self.num_batches = 0
        self.total_loss = torch.tensor(0.0, device=pl_module.device)
        if self.ema is not None and isinstance(self.ema, torch.Tensor):
            self.ema = self.ema.to(pl_module.device)

    def on_validation_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            **kwargs
    ) -> None:
        """
        Called when a validation batch ends.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
            outputs (Any): The outputs from the validation step.
            batch (Any): The input batch.
            batch_idx (int): The index of the current batch.
            **kwargs: Additional keyword arguments.
        """
        self.total_loss += outputs
        self.num_batches += 1

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Called when the validation epoch ends.

        Args:
            trainer (L.Trainer): The trainer instance.
            pl_module (L.LightningModule): The LightningModule instance.
        """
        avg_loss = self.total_loss / self.num_batches
        if self.ema is None:
            self.ema = avg_loss
        else:
            if self.soft_beta is not None:
                avg_loss = torch.min(torch.stack([avg_loss, self.ema * self.soft_beta]))
            self.ema = self.alpha * self.ema + (1 - self.alpha) * avg_loss
        pl_module.log(self.ema_log_name, self.ema, on_step=False, on_epoch=True)
