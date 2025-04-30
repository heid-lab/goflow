import pickle
from pathlib import Path
from typing import Optional

import lightning as L
from torch_geometric.loader import DataLoader
from gotennet.data.components.utils import CountNodesPerGraph, ConformationDataset


class RDB7DataModule(L.LightningDataModule):
    def __init__(self, data_file: str, split_path: str, split_file: str, batch_size: int = 64):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.transforms = CountNodesPerGraph()
        
        # Load train, val, test split indices from the pickle file
        with open(Path(split_path) / split_file, "rb") as f:
            self.split_dict = pickle.load(f)
            
        self.train_indices = self.split_dict["train"]
        self.val_indices = self.split_dict["val"]
        self.test_indices = self.split_dict["test"]

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        train_set = ConformationDataset(
            self.data_file, 
            data_indices=self.train_indices, 
            transform=self.transforms
        )
        return DataLoader(
            train_set, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=2
        )

    def val_dataloader(self) -> DataLoader:
        val_set = ConformationDataset(
            self.data_file, 
            data_indices=self.val_indices, 
            transform=self.transforms
        )
        return DataLoader(
            val_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )

    def test_dataloader(self) -> DataLoader:
        test_set = ConformationDataset(
            self.data_file, 
            data_indices=self.test_indices, 
            transform=self.transforms
        )
        return DataLoader(
            test_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )