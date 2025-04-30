import pandas as pd
import pickle
import os
import numpy as np
import torch
import tqdm
import argparse
from utils.datasets import generate_ts_data2
from ase.io import iread


def parse_xyz_corpus_ase(filename):
    return [atoms.positions for atoms in iread(filename)]


def data_processing_main(args):
    # Parse XYZ file
    if args.only_sampling == 0:
        xyz_blocks = parse_xyz_corpus_ase(args.xyz_file)
        # put r/ts/p into one np.array => one rxn
        rxn_block_B_RTSP_N_3D = [np.array(xyz_blocks[i:i + 3]) for i in range(0, len(xyz_blocks), 3)]
    
    # Read CSV file with reaction data
    df = pd.read_csv(args.csv_file)
    rxn_smarts = df.smiles
    rxn_indices = df.rxn
    
    if args.with_energies:
        energies_r_ts_p_G_3 = df[['r_energy', 'ts_energy', 'p_energy']].values
        energies_std_r_ts_p_G_3 = (energies_r_ts_p_G_3 - energies_r_ts_p_G_3.mean(
            axis=0)) / energies_r_ts_p_G_3.std(axis=0)

    if args.only_sampling == 1:
        rxn_block_B_RTSP_N_3D = [None] * len(rxn_smarts)

    # set feature types
    # if there exist pre-defined feat_dict, load the feat_dict
    if os.path.isfile(args.feat_dict):
        feat_dict = pickle.load(open(args.feat_dict, "rb"))
    else:
        print(args.feat_dict, "is not exist. Use default feat_dict.")
        feat_dict = {
            "GetIsAromatic": {},
            "GetFormalCharge": {},
            "GetHybridization": {},
            "GetTotalNumHs": {},
            "GetTotalValence": {},
            "GetTotalDegree": {},
            "GetChiralTag": {},
            "IsInRing": {},
        }

    # generate torch_geometric.data.Data instance
    data_list = []

    if args.with_energies:
        for (rxn_idx, a_smarts, energies_std_r_ts_p_3, rxn_block_RTSP_N_3D) in tqdm.tqdm(
                zip(rxn_indices, rxn_smarts, energies_std_r_ts_p_G_3, rxn_block_B_RTSP_N_3D)):
            feat_dict = add_rxn_datapoint(args, feat_dict, data_list, rxn_idx, a_smarts,
                                          rxn_block=rxn_block_RTSP_N_3D, energies=energies_std_r_ts_p_3)
    else:
        for (rxn_idx, a_smarts, rxn_block_RTSP_N_3D) in tqdm.tqdm(
                zip(rxn_indices, rxn_smarts, rxn_block_B_RTSP_N_3D)):
            feat_dict = add_rxn_datapoint(args, feat_dict, data_list, rxn_idx, a_smarts,
                                          rxn_block=rxn_block_RTSP_N_3D, energies=None)

    # convert features to one-hot encoding
    num_cls = [len(v) for k, v in feat_dict.items()]
    for data in data_list:
        feat_onehot = []
        feats = data.r_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.r_feat = torch.cat(feat_onehot, dim=-1)

        feat_onehot = []
        feats = data.p_feat.T
        for feat, n_cls in zip(feats, num_cls):
            feat_onehot.append(torch.nn.functional.one_hot(feat, num_classes=n_cls))
        data.p_feat = torch.cat(feat_onehot, dim=-1)

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save single data file
    with open(os.path.join(args.save_dir, "data.pkl"), "wb") as f:
        pickle.dump(data_list, f)
    with open(os.path.join(args.save_dir, "feat_dict.pkl"), "wb") as f:
        pickle.dump(feat_dict, f)


def add_rxn_datapoint(args, feat_dict: dict, data_list: list, idx: int, a_smarts: str, xyz_block=None, rxn_block=None,
                      energies=None):
    r, p = a_smarts.split(">>")
    data, feat_dict = generate_ts_data2(r, p,
                                      energies=energies,
                                      xyz_block=xyz_block,
                                      rxn_block=rxn_block,
                                      feat_dict=feat_dict,
                                      only_sampling=bool(args.only_sampling)
                                      )
    data_list.append(data)
    data.augmented = False
    data.rxn_index = idx
    return feat_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to rdb7_full.csv")
    parser.add_argument("--xyz_file", type=str, required=True, help="Path to rdb7_full.xyz")
    parser.add_argument("--only_sampling", type=int, default=0,
                        help="use data only for sampling R/TS/P (no 3D positions given)")
    parser.add_argument("--feat_dict", type=str, default="xxx.pkl", help="path/to/feat_dict.pkl")
    parser.add_argument("--save_dir", type=str, default="data/processed")
    parser.add_argument("--with_energies", action="store_true", default=False)

    args = parser.parse_args()
    data_processing_main(args)
