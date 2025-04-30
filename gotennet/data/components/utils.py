import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
import pickle
from torch_geometric.data import Dataset


class ConformationDataset(Dataset):
    def __init__(self, data_file, data_indices, transform=None):
        """
        Select data according to rxn_index attribute.
        
        Parameters:
        - data_file: Path to the pickle file containing all data samples
        - data_indices: Indices matching the rxn_index attribute in data objects
        - transform: Optional transforms to apply to each data sample
        """
        super().__init__()
        
        with open(data_file, "rb") as f:
            all_data = pickle.load(f)
        
        # Create a mapping from rxn_index to the data object
        rxn_index_to_data = {}
        for data_obj in all_data:
            rxn_index_to_data[data_obj.rxn_index] = data_obj
        
        # Get data by the provided rxn_indices
        self.data = [rxn_index_to_data[idx] for idx in data_indices]
        self.transform = transform
        
        # Cache atom and edge types
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
    
    
class CountNodesPerGraph(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        if not hasattr(data, "__num_nodes__"):
            data.num_nodes = len(data.pos)
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data


class MissingLabelException(Exception):
    pass
