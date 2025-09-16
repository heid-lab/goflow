import pickle 
from gotennet.models.components.callbacks import evaluate_geometry
from ase import Atoms
from ase.io import write
import torch
from pathlib import Path
import random
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def compute_stats_for_samples_file(samples_file, out_folder):
    with open(samples_file, 'rb') as f:
        samples_all = pickle.load(f)

    best_maes = []
    median_maes = []
    random_maes = []

    best_rmses = []
    median_rmses = []
    random_rmses = []

    best_angle_errors = []
    median_angle_errors = []
    random_angle_errors = []

    # Inference time per reaction (already averaged per rxn)
    inference_times = []

    per_rxn_rows = []

    for data in tqdm(samples_all):        
        if not hasattr(data, 'pos_gen_all_samples_S_N_3'):
            if len(data.pos_gen.shape) == 2:
                data.pos_gen = data.pos_gen.unsqueeze(0)
            data.pos_gen_all_samples_S_N_3 = data.pos_gen

        curr_rxn = data.rxn_index.item()
                
        # --- Evaluate median sample geometry ---
        res_med = evaluate_geometry(data)
        mae_med = res_med['mae']
        rmse_med = res_med['rmse']
        angle_med = res_med['angle_error']

        # --- Per-reaction avg inference time ---
        avg_inf_time = getattr(data, 'avg_inference_time', None)
        if avg_inf_time is not None:
            inference_times.append(avg_inf_time)

        # --- Evaluate all generated samples ---
        data_cp = data.clone()
        mae_samples_S, rmse_samples_S, angle_samples_S = [], [], []
        
        num_samples = len(data.pos_gen_all_samples_S_N_3)
        for s_i in range(num_samples):
            data_cp.pos_gen = [data.pos_gen_all_samples_S_N_3[s_i]]
            res_sample = evaluate_geometry(data_cp)
            
            mae_samples_S.append(res_sample['mae'])
            rmse_samples_S.append(res_sample['rmse'])
            angle_samples_S.append(res_sample['angle_error'])

        # --- Find best sample ---
        min_mae = min(mae_samples_S)
        min_rmse = min(rmse_samples_S)
        min_angle = min(angle_samples_S)

        # --- Pick a random sample ---
        random.seed(42 + curr_rxn)  # deterministic per reaction
        rand_idx = random.randrange(num_samples)
        mae_rand = mae_samples_S[rand_idx]
        rmse_rand = rmse_samples_S[rand_idx]
        angle_rand = angle_samples_S[rand_idx]

        # --- Accumulate MAEs for summary ---
        best_maes.append(min_mae)
        median_maes.append(mae_med)
        random_maes.append(mae_rand)
        
        best_rmses.append(min_rmse)
        median_rmses.append(rmse_med)
        random_rmses.append(rmse_rand)
        
        best_angle_errors.append(min_angle)
        median_angle_errors.append(angle_med)
        random_angle_errors.append(angle_rand)

        per_rxn_rows.append({
            'rxn_id': curr_rxn,
            
            'mae_best': min_mae,
            'mae_median': mae_med,
            'mae_random': mae_rand,

            'rmse_best': min_rmse,
            'rmse_median': rmse_med,
            'rmse_random': rmse_rand,

            'angle_error_best': min_angle,
            'angle_error_median': angle_med,
            'angle_error_random': angle_rand,

            'avg_inference_time': avg_inf_time,
        })

    # --- After processing all reactions: write summaries ---
    summary_dir = Path("reaction_analysis") / out_folder
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Per-reaction CSV
    per_rxn_df = pd.DataFrame(per_rxn_rows)
    per_rxn_df.sort_values('rxn_id').to_csv(summary_dir / 'stats_per_reaction.csv', index=False, float_format='%.6f')

    combined_summary_rows = [
        {
            'category': 'best_of_all',
            'mean_mae': statistics.mean(best_maes),
            'mean_rmse': statistics.mean(best_rmses),
            'mean_angle_error': statistics.mean(best_angle_errors),
            'n': len(best_maes),
        },
        {
            'category': 'median_sample',
            'mean_mae': statistics.mean(median_maes),
            'mean_rmse': statistics.mean(median_rmses),
            'mean_angle_error': statistics.mean(median_angle_errors),
            'n': len(median_maes),
        },
        {
            'category': 'random_of_all',
            'mean_mae': statistics.mean(random_maes),
            'mean_rmse': statistics.mean(random_rmses),
            'mean_angle_error': statistics.mean(random_angle_errors),
            'n': len(random_maes),
        },
    ]

    pd.DataFrame(combined_summary_rows).to_csv(summary_dir / 'summary_means.csv', index=False, float_format='%.6f')
    print("Mean summary (MAE, RMSE, Angle Error):")
    for row in combined_summary_rows:
        print(
            f"  {row['category']}: mean MAE = {row['mean_mae']:.6f}, "
            f"mean RMSE = {row['mean_rmse']:.6f}, "
            f"mean Angle Error = {row['mean_angle_error']:.6f} over n={row['n']}"
        )

    # --- Inference time summary ---
    print("Inference time summary:")
    print(f"  Collected {len(inference_times)} avg_inference_time values.")
    if len(inference_times) > 0:
        mean_inference_time = statistics.mean(inference_times)
        pd.DataFrame([{
            'mean_avg_inference_time': mean_inference_time,
            'n': len(inference_times)
        }]).to_csv(summary_dir / 'summary_inference_time.csv', index=False, float_format='%.6f')
        print(f"Mean avg_inference_time = {mean_inference_time:.6f} over n={len(inference_times)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute stats for saved samples file")
    parser.add_argument("samples_file", nargs='?', default='xxx', help='Path to the pickled samples file')
    parser.add_argument("out_folder", nargs='?', default='xxx', help='Output folder under reaction_analysis')
    args = parser.parse_args()

    compute_stats_for_samples_file(args.samples_file, args.out_folder)
