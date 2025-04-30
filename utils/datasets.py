import pickle
import copy

import numpy as np

import torch
from torch_geometric.data import Data, Dataset

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def read_xyz_block(xyz_block):
    sxyz = xyz_block.split("\n")[2:]
    if sxyz[-1]:
        pass
    else:
        sxyz = sxyz[:-1]

    symbols = []
    pos = []
    for line in sxyz:
        _ = line.split("\t")
        symbols.append(_[0])
        pos.append([float(u) for u in _[1:]])

    symbols = np.array(symbols)
    pos = np.array(pos)
    return symbols, pos


def generate_ts_data2(
    r_smarts,
    p_smarts,
    energies=None,
    xyz_block=None,
    rxn_block=None,
    data_cls=Data,
    feat_dict={},
    only_sampling=True,
):
    if isinstance(r_smarts, str) and isinstance(p_smarts, str):
        r = Chem.MolFromSmarts(r_smarts)
        p = Chem.MolFromSmarts(p_smarts)
        Chem.SanitizeMol(r)
        Chem.SanitizeMol(p)
    else:
        r, p = r_smarts, p_smarts
    N = r.GetNumAtoms()
    if xyz_block is not None:
        if isinstance(xyz_block, str):
            symbol_xyz, pos = read_xyz_block(xyz_block)
        else:
            pos = xyz_block
        pos = torch.Tensor(pos)
        assert len(pos) == N and p.GetNumAtoms() == N
    elif rxn_block is not None:
        pos = torch.Tensor(rxn_block) # rxn_block_RTSP_N_3D
        pos = pos.transpose(0, 1)
        assert pos.shape[0] == N and p.GetNumAtoms() == N
    else:
        pos = torch.zeros(N,3)

    r_perm = np.array([a.GetAtomMapNum() for a in r.GetAtoms()]) - 1
    p_perm = np.array([a.GetAtomMapNum() for a in p.GetAtoms()]) - 1
    r_perm_inv = np.argsort(r_perm)
    p_perm_inv = np.argsort(p_perm)

    r_chiraltype = []
    r_atomic_number = []
    r_feat = []

    p_chiraltype = []
    p_atomic_number = []
    p_feat = []

    def get_closest_value(v, feat):
        return v.get(
            feat,
            v[min(v.keys(), key=lambda k: abs(int(k) - int(feat)))] # get closest key to 'feat' in 'v', return its value
        )

    # feat: len(v) done for one-hot encoding of feat based on len(v)
    for atom in np.array(r.GetAtoms())[r_perm_inv]:
        r_chiraltype.append(atom.GetChiralTag())
        r_atomic_number.append(atom.GetAtomicNum())
        atomic_feat = []
        for k, v in feat_dict.items():
            feat = getattr(atom, k)()
            if not only_sampling:
                if feat not in v:
                    v.update({feat: len(v)})
            atomic_feat.append(get_closest_value(v, feat))
        r_feat.append(atomic_feat)

    for atom in np.array(p.GetAtoms())[p_perm_inv]:
        p_chiraltype.append(atom.GetChiralTag())
        p_atomic_number.append(atom.GetAtomicNum())
        atomic_feat = []
        for k, v in feat_dict.items():
            feat = getattr(atom, k)()
            if not only_sampling:
                if feat not in v:
                    v.update({feat: len(v)})
            atomic_feat.append(get_closest_value(v, feat))
        p_feat.append(atomic_feat)

    assert r_atomic_number == p_atomic_number
    z = torch.tensor(r_atomic_number, dtype=torch.long)
    r_feat = torch.tensor(r_feat, dtype=torch.long)
    p_feat = torch.tensor(p_feat, dtype=torch.long)
    r_adj = Chem.rdmolops.GetAdjacencyMatrix(r)
    p_adj = Chem.rdmolops.GetAdjacencyMatrix(p)
    r_adj_perm = r_adj[r_perm_inv, :].T[r_perm_inv, :].T
    p_adj_perm = p_adj[p_perm_inv, :].T[p_perm_inv, :].T
    adj = r_adj_perm + p_adj_perm
    row, col = adj.nonzero()

    _nonbond = 0
    p_edge_type = []
    for i, j in zip(p_perm_inv[row], p_perm_inv[col]):
        b = p.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            p_edge_type.append(BOND_TYPES[b.GetBondType()])
        elif b is None:
            p_edge_type.append(_nonbond)

    r_edge_type = []
    for i, j in zip(r_perm_inv[row], r_perm_inv[col]):
        b = r.GetBondBetweenAtoms(int(i), int(j))
        if b is not None:
            r_edge_type.append(BOND_TYPES[b.GetBondType()])
        elif b is None:
            r_edge_type.append(_nonbond)

    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    
    r_nonzero = np.array(r_adj_perm.nonzero())
    r_edge_index = torch.tensor(r_nonzero, dtype=torch.long)
    
    p_nonzero = np.array(p_adj_perm.nonzero())
    p_edge_index = torch.tensor(p_nonzero, dtype=torch.long)

    r_edge_type = torch.tensor(r_edge_type)
    p_edge_type = torch.tensor(p_edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    r_perm_tensor = (r_edge_index[0] * N + r_edge_index[1]).argsort()
    p_perm_tensor = (p_edge_index[0] * N + p_edge_index[1]).argsort()

    edge_index = edge_index[:, perm]
    r_edge_index = r_edge_index[:, r_perm_tensor]
    p_edge_index = p_edge_index[:, p_perm_tensor]

    r_edge_type = r_edge_type[perm]
    p_edge_type = p_edge_type[perm]

    smiles = f"{r_smarts}>>{p_smarts}"
    # edge_type = torch.stack([r_edge_type, p_edge_type]).T
    edge_type = r_edge_type * len(BOND_TYPES) + p_edge_type
    
    r_perm_tensor = torch.from_numpy(r_perm)
    p_perm_tensor = torch.from_numpy(p_perm)
    r_perm_inv_tensor = torch.from_numpy(r_perm_inv)
    p_perm_inv_tensor = torch.from_numpy(p_perm_inv)
    
    # Convert chiraltypes to tensor too
    r_chiraltype_tensor = torch.tensor(r_chiraltype)
    p_chiraltype_tensor = torch.tensor(p_chiraltype)
    
    # Handle energies
    energies_tensor = torch.tensor(energies, dtype=torch.float32) if energies is not None else None

    data = data_cls(
        atom_type=z,
        r_feat=r_feat,
        p_feat=p_feat,
        r_chiraltype=r_chiraltype_tensor,
        p_chiraltype=p_chiraltype_tensor,
        pos=pos,

        edge_index=edge_index,
        r_edge_index=r_edge_index,
        p_edge_index=p_edge_index,
        edge_type=edge_type,
        rdmol=(copy.deepcopy(r), copy.deepcopy(p)),
        smiles=smiles,
        energies=energies_tensor,

        r_index_to_mapnum=r_perm_tensor,
        p_index_to_mapnum=p_perm_tensor,
        mapnum_to_r_index=r_perm_inv_tensor,
        mapnum_to_p_index=p_perm_inv_tensor
    )
    return data, feat_dict


class TSDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform

    def get(self, idx):
        return self.data[idx]

    def len(self):
        return len(self.data)


class ConformationDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):
        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data


    def __len__(self):
        return len(self.data)

    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)
