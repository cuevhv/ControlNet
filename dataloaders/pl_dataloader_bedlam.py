import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from dataloaders.dataset_bedlam import BedlamSimpleDataset
import os


class BedlamDataModule(pl.LightningDataModule):
    def __init__(self, train_data_json: str, val_data_json: str, test_data_json: str = None,
                 control_type = "segment_human_and_clothes", train_dataloader_conf: Optional[dict] = None):
        super().__init__()

        self.train_data_json = train_data_json
        self.val_data_json = val_data_json
        self.test_data_json = test_data_json
        self.control_type = control_type

        self.train_dataloader_conf: Optional[dict] = train_dataloader_conf


    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train = BedlamSimpleDataset(self.train_data_json, condition_type=self.control_type)

            if self.val_data_json is not None:
                self.val = BedlamSimpleDataset(self.val_data_json, condition_type=self.control_type)

        if (self.test_data_json is not None and stage == "test"):
            self.test = BedlamSimpleDataset(self.test_data_json, condition_type=self.control_type)

        if self.trainer.is_global_zero:
            print("Training dataset length:", len(self.train))
            print("Validation dataset length:", len(self.val) if self.val_data_json is not None \
                                                        else "No validation dataset")


    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_conf)


    def val_dataloader(self):
        if self.val_data_json is None:
            return None
        return DataLoader(self.val, **self.train_dataloader_conf)


    def test_dataloader(self):
        if self.test_data_json is None:
            return None
        return DataLoader(self.test, **self.train_dataloader_conf)